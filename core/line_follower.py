#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉循迹学生实验版。

基础实验需完成 TODO-B1 ~ TODO-B7。教师已保留相机、串口、形态学、
连通域过滤、调试显示和退出停车代码；学生只补视觉与控制核心。

重要安全约定：未补全时程序能够启动，但所有占位逻辑均导向停车，
不会默认驱动车辆。首次实车测试必须架空驱动轮或将车放在宽阔区域。

提高实验只给任务要求，不提供算法或状态机骨架：

  A1 十字路口：在黑胶带十字路口前可靠识别，按学生自定策略完成
     直行、左转或右转；进入路口必须降速，决策或重捕获超时必须停车。
  A2 左/右直角转弯：分别完成 90° 左转和右转，转弯前降速，转弯后
     重新稳定捕获线路；识别失败、超时、相机或串口异常必须停车。

底盘输出约定：x 为前进速度(mm/s)，z 为转向角速度(mrad/s)。
"""
import logging
import math
import threading
import time
from itertools import combinations

import cv2
import numpy as np

from core.odometry import ImuOdometry

logger = logging.getLogger(__name__)


# =====================================================================
# 1. 线条检测器（追踪式）
# =====================================================================
class LineDetector:
    """从画面中提取巡线路径中心与方向（处理在降采样小图上进行）。"""

    def __init__(self, work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, min_seg_width=2,
                 polarity='black',
                 crop_bottom_frac=0.70, crop_top_frac=0.90,
                 track_half=50.0, scan_start_ratio=0.25,
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0,
                 line_width_model=None, enforce_width=True):
        self.work_width = work_width
        self.roi_top_ratio = roi_top_ratio          # 垂直方向：只处理底部这段(车前方地面)
        self.n_scan_rows = n_scan_rows              # 扫描行数
        self.min_seg_width = min_seg_width          # 过滤过窄噪点段
        self.polarity = polarity                    # 'black'=黑线白底 / 'white'=白线黑底
        # 梯形裁切：底部保留比例 / 顶部保留比例（相对整幅宽）
        self.crop_bottom_frac = crop_bottom_frac    # 底部(近车头)窗口窄
        self.crop_top_frac = crop_top_frac          # 顶部(远处)窗口宽，留转弯余量
        self.track_half = track_half                # 滑动搜索窗半宽(px, 工作图坐标)
        self.scan_start_ratio = scan_start_ratio    # 扫描起点在 ROI 内的比例(0~1)
        self.binary_mode = binary_mode
        self.fixed_threshold = int(np.clip(fixed_threshold, 0, 255))
        self.adaptive_block = max(3, int(adaptive_block) | 1)
        self.adaptive_c = float(adaptive_c)
        self.line_width_model = self._validate_line_width_model(
            line_width_model)
        self.enforce_width = bool(enforce_width)

        # 上一帧车头参考行处的线中心(工作图 x)，兼作本帧搜索窗中心
        self._prev_cx = None
        # Only the opt-in traffic controller requests branch selection.
        self.path_preference = None
        # A left/right road becomes forward-facing after the vehicle enters it.
        # Remember that capture so its per-frame label can hand off to straight.
        self._captured_turn = None

    @staticmethod
    def _validate_line_width_model(model):
        if model is None:
            return None
        clean = {name: float(model[name]) for name in (
            'horizontal_fov_deg', 'camera_height_m', 'pitch_down_deg',
            'segmentation_scale', 'min_width_mm', 'max_width_mm')}
        if not 1.0 < clean['horizontal_fov_deg'] < 179.0:
            raise ValueError('horizontal_fov_deg must be between 1 and 179')
        if clean['camera_height_m'] <= 0 or clean['segmentation_scale'] <= 0:
            raise ValueError('camera height and segmentation scale must be positive')
        if not 0 < clean['min_width_mm'] < clean['max_width_mm']:
            raise ValueError('line width range must be positive and ordered')
        return clean

    def _physical_line_width_mm(self, pixel_width, pixel_y, image_height):
        """Convert a horizontal binary run to calibrated ground width."""
        model = self.line_width_model
        if model is None:
            return None
        cx = (self.work_width - 1) / 2.0
        cy = (float(image_height) - 1) / 2.0
        focal = cx / math.tan(math.radians(
            model['horizontal_fov_deg']) / 2.0)
        down_ray = (float(pixel_y) - cy) / focal
        pitch = math.radians(model['pitch_down_deg'])
        ray_down = math.sin(pitch) + down_ray * math.cos(pitch)
        if ray_down <= 1e-6:
            return None
        metres_per_pixel = model['camera_height_m'] / (ray_down * focal)
        return (float(pixel_width) * metres_per_pixel * 1000.0 *
                model['segmentation_scale'])

    # ------------------------------------------------------------------
    def process(self, frame):
        """处理一帧，返回检测结果 dict。"""
        empty = self._empty_result()
        if self.path_preference not in ('left', 'right'):
            self._captured_turn = None
        elif self._captured_turn not in (None, self.path_preference):
            self._captured_turn = None
        if frame is None or frame.size == 0:
            return empty

        h, w = frame.shape[:2]
        scale = self.work_width / w
        work = cv2.resize(frame, (self.work_width, int(h * scale)))
        wh, ww = work.shape[:2]

        roi_top = int(wh * self.roi_top_ratio)
        roi = work[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 10:
            return empty

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # TODO-B1【梯形 ROI 掩膜】
        # 目标：生成与 blur 同形状的 bool 数组 inside。每一行只保留画面
        # 中央的一段；ROI 顶部(远处)宽度比例为 crop_top_frac，底部
        # (近处)为 crop_bottom_frac，中间按行线性变化。
        # 提示：
        #   1) frac = np.linspace(顶部比例, 底部比例, roi_h)
        #   2) half = frac * ww * 0.5，center = ww / 2.0
        #   3) 利用 cols[None, :] 与左右边界[:, None]广播比较
        # 验收：inside.dtype 为 bool、shape == blur.shape，且上下宽度符合参数。
        center = ww / 2.0
        top_frac = float(np.clip(self.crop_top_frac, 0.0, 1.0))
        bottom_frac = float(np.clip(self.crop_bottom_frac, 0.0, 1.0))
        fractions = np.linspace(top_frac, bottom_frac, roi_h)
        half_widths = fractions * ww * 0.5
        columns = np.arange(ww, dtype=np.float64)[None, :]
        inside = ((columns >= (center - half_widths)[:, None]) &
                  (columns < (center + half_widths)[:, None]))

        # TODO-B2【二值化】
        # fixed：使用 fixed_threshold；otsu：调用学生手写的 _otsu；
        # adaptive：调用 _adaptive_binary（基础拓展，可选）。
        # 无论哪种模式，都只能在 inside 内产生 255，窗外必须保持 0。
        in_vals = blur[inside]
        if in_vals.size < 50:
            return self._empty_result(binary=np.zeros_like(blur), roi_top=roi_top)

        if self.binary_mode == 'adaptive':
            binary = self._adaptive_binary(blur, inside)
        else:
            if self.binary_mode == 'fixed':
                threshold = self.fixed_threshold
            else:
                threshold = self._otsu(in_vals)
            # 退化保护必须保留：近乎纯黑/纯白画面不得误判为线路。
            if threshold <= 5 or threshold >= 250:
                self._prev_cx = None
                return self._empty_result(binary=None, roi_top=roi_top)
            binary = self._apply_global_threshold(blur, inside, threshold)

        # 前景占比校验：一条线只应占窗内很小比例，占满整窗说明是假检(纯色/大面积暗区)
        fg_ratio = binary[inside].mean() / 255.0
        if fg_ratio < 0.005 or fg_ratio > 0.35:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # Branch discovery gets a wider trapezoid than ordinary tracking. The
        # normal mask stays narrow for noise rejection; only candidates that
        # reconnect to the near stem are accepted from this wider view.
        branch_fractions = np.linspace(max(top_frac, 0.90),
                                       max(bottom_frac, 0.70), roi_h)
        branch_half = branch_fractions * ww * 0.5
        branch_inside = ((columns >= (center-branch_half)[:, None]) &
                         (columns < (center+branch_half)[:, None]))
        if self.binary_mode == 'adaptive':
            branch_binary = self._adaptive_binary(blur, branch_inside)
        else:
            branch_binary = self._apply_global_threshold(
                blur, branch_inside, threshold)
        branch_binary = cv2.morphologyEx(branch_binary, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(
                                             cv2.MORPH_RECT, (3, 5)))
        _, branch_labels, branch_stats, _ = cv2.connectedComponentsWithStats(
            branch_binary, connectivity=8)
        branch_mask = np.zeros_like(branch_binary)
        for index in range(1, branch_labels.max()+1):
            _, _, bw_, bh_, area = branch_stats[index]
            if area < 25 or bh_ < roi_h*0.20:
                continue
            fill = area / float(bw_*bh_)
            if bw_ > 8 and bh_ > 8 and fill > 0.85:
                continue
            branch_mask[branch_labels == index] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # --- 3. 连通域几何过滤：保留"条带状"线分量（对曲线弯更宽容） ---
        # 一次连通的黑线在图像上是条带。过滤原则：
        #   * 剔除明显“矮又宽”的横条(纯色噪点/大色块) → 高度占比过低时丢弃
        #   * 允许曲线弯处变宽/变横的长条带通过(不再硬性要求纵向细长)
        # 用 高度占比 + 面积/包围盒 挤出比例 来判断，曲线弯(横向但有长度)也能保留。
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        line_mask = np.zeros_like(binary)
        min_h = roi_h * 0.25          # 线至少要覆盖 ROI 25% 的高度(曲线弯放宽)
        for i in range(1, labels.max() + 1):
            x_, y_, bw_, bh_, area = stats[i]
            if area < 25:
                continue
            # 纵向“细长”：高占比足够 且 不被横向色块挤爆
            if bh_ < min_h:
                continue
            # 面积极度低→细带；面积高但相对包围盒仍细长(弯段绕行)→也接受
            fill = area / float(bw_ * bh_)          # 0~1，实心色块接近1
            if bw_ > 8 and bh_ > 8 and fill > 0.85:
                continue                            # 实心大色块→剔除(防假检)
            line_mask[labels == i] = 255
        binary = line_mask
        if int(binary.max()) == 0:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # L 弯不能由每行一个中心点的多项式可靠表达。先在完整连通域上
        # 寻找“纵向主干 + 单侧长横臂”，把方向和拐点位置交给控制状态机。
        corner = self._detect_l_corner(binary, roi_top)

        # --- 4. 扫描线 + 滑动搜索窗 ---
        # 预测位置作为搜索窗中心；丢线后重捕获时清空预测窗
        pred = self._prev_cx if self._prev_cx is not None else center
        # Branch selection needs a neutral incoming-stem fit first. Once all
        # candidates are built below, the requested left/right path replaces it.
        baseline_preference = ('continuation' if self.path_preference in
                               ('left', 'straight', 'right') else None)
        points = self._scan_lines(binary, roi_top, ww, inside, pred,
                                  baseline_preference)
        if len(points) < 3:
            # 重捕获：不受预测限制，直接在全裁切范围内找
            points = self._scan_lines(binary, roi_top, ww, inside, None,
                                      baseline_preference)
        if len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        if corner.get('junction_straight') and self.path_preference is None:
            # 交叉点的横臂会成为扫描行里的最长黑段并把拟合中心拉向支路。
            # 只保留贴近贯穿主干的点，让车辆沿进入路口时的主线直行。
            stem_x = float(corner['junction_stem_x'])
            stem_tol = max(14.0, float(corner['junction_normal_width']) * 1.5)
            stem_points = [point for point in points
                           if abs(float(point[0]) - stem_x) <= stem_tol]
            if len(stem_points) >= 3:
                points = stem_points

        # 地缝、反光边缘常只贡献一两个远离主线的扫描点。先选取横向
        # 位置连续的最长点链，再做稳健二次拟合，避免端点把曲线拉飞。
        points = self._select_continuous_path(points)
        if len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        fit_coeffs, points = self._robust_quadratic_fit(points)
        if fit_coeffs is None or len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # 道路线必须延伸到近车头区域。只在 ROI 中上部出现的细长物体
        # （例如电线）即使能提供多个扫描点，也不能向底部外推成道路。
        near_y = roi_top + int(roi_h * 0.80)
        near_points = [point for point in points if point[1] >= near_y]
        if len(near_points) < 2:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # A drivable line must enter the near field as a narrow ribbon.  Floor
        # shadows can survive Otsu and connected-component filtering as a
        # tall, porous shape; their scan-line widths remain far larger than a
        # perspective-expanded tape line.  Use the median so a real junction
        # may contain one or two wide rows while its incoming stem stays valid.
        near_widths = np.asarray([point[2] for point in near_points],
                                 dtype=np.float64)
        line_width_mm = None
        if self.line_width_model is not None:
            physical_widths = [
                self._physical_line_width_mm(point[2], point[1], wh)
                for point in near_points]
            physical_widths = [width for width in physical_widths
                               if width is not None and math.isfinite(width)]
            if not physical_widths:
                if self.enforce_width:
                    self._prev_cx = None
                    return self._empty_result(binary=binary, roi_top=roi_top)
            else:
                line_width_mm = float(np.median(physical_widths))
                if (self.enforce_width and not
                        (self.line_width_model['min_width_mm'] <= line_width_mm <=
                         self.line_width_model['max_width_mm'])):
                    self._prev_cx = None
                    return self._empty_result(binary=binary, roi_top=roi_top)
            # Physical width is diagnostic when enforcement is disabled.
        else:
            near_width_limit = max(25.0, ww * 0.08)
            if float(np.median(near_widths)) > near_width_limit:
                self._prev_cx = None
                return self._empty_result(binary=binary, roi_top=roi_top)

        split_candidates = self._detect_split_branches(
            branch_mask, roi_top, ww, fit_coeffs, points)
        branch_candidates = (split_candidates or
                             self._branch_candidates(corner, ww))
        chosen = None
        if split_candidates:
            directions = {item['direction'] for item in split_candidates}
            split_y = max(item.get('split_y', roi_top)
                          for item in split_candidates)
            targets = [item['target_x'] for item in split_candidates]
            corner = dict(corner)
            corner.update({
                'corner_point': (float(np.polyval(fit_coeffs, split_y)),
                                 float(split_y)),
                'corner_y_ratio': float((split_y-roi_top) /
                                        max(1, roi_h-1)),
                'corner_span': float(max(targets)-min(targets)),
                'junction_left': 'left' in directions,
                'junction_straight': 'straight' in directions,
                'junction_right': 'right' in directions,
                'junction_near': bool((split_y-roi_top) /
                                      max(1, roi_h-1) >= 0.60),
                'junction_stem_x': float(np.polyval(fit_coeffs, split_y)),
            })
            requested = self.path_preference
            if requested == 'continuation':
                requested = 'straight'
            chosen = next((item for item in split_candidates
                           if item['direction'] == requested), None)
            if chosen is not None and requested in ('left', 'right'):
                self._captured_turn = requested
            if chosen is not None and chosen['direction'] != 'straight':
                fit_coeffs = None
                points = list(chosen['points'])

        # TODO-B4【二次拟合、横向误差与方向角】
        # 用 x = q2*y^2 + q1*y + q0 描述平滑弯道；车头参考点取
        # ROI 最底行，方向角取该点切线 dx/dy = 2*q2*y + q1。
        ref_y = roi_top + roi_h - 1
        if chosen is not None and chosen['direction'] != 'straight':
            error_px = float(chosen['error_px'])
            cx_fit = float(np.clip(ww/2.0 + error_px, 0.0, ww-1.0))
            angle_deg = float(chosen['angle_deg'])
        else:
            q2, q1, q0 = fit_coeffs
            cx_fit = float(np.clip(q2*ref_y**2 + q1*ref_y + q0,
                                   0.0, ww-1.0))
            error_px = cx_fit - ww/2.0
            tangent = 2.0*q2*ref_y + q1
            angle_deg = math.degrees(math.atan(-tangent))
        if self._prev_cx is None:
            self._prev_cx = cx_fit
        else:
            self._prev_cx = 0.6 * cx_fit + 0.4 * self._prev_cx

        far_limit = roi_top + int(roi_h * 0.45)
        near_limit = roi_top + int(roi_h * 0.75)
        far_points = sum(1 for point in points if point[1] <= far_limit)
        close_points = sum(1 for point in points if point[1] >= near_limit)
        # This is only a per-frame candidate. TrafficBehavior additionally
        # requires three frames and matching forward odometry before treating
        # it as the natural end of the cross branch.
        line_end_candidate = close_points >= 2 and far_points <= 1

        return {
            'is_valid': True,
            'centroid': (cx_fit, float(ref_y)),
            'error_px': float(error_px),          # 线在右 → 正 → 右转
            'angle_deg': float(angle_deg),
            'line_width_mm': line_width_mm,
            'fit_coeffs': (None if fit_coeffs is None else
                           tuple(float(value) for value in fit_coeffs)),
            **corner,
            'branch_candidates': branch_candidates,
            'selected_branch_direction': (None if chosen is None else
                                          chosen['direction']),
            'line_end_candidate': bool(line_end_candidate),
            'points': points,                   # 参与拟合的点
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
        }

    def _select_continuous_path(self, points):
        """保留横向连续的最长扫描点链，剔除跳到地缝/反光边缘的点。"""
        if len(points) < 3:
            return list(points)
        ordered = sorted(points, key=lambda point: point[1])
        max_step = max(18.0, min(45.0, self.track_half * 0.65))
        runs = []
        current = [ordered[0]]
        for point in ordered[1:]:
            if abs(float(point[0]) - float(current[-1][0])) <= max_step:
                current.append(point)
            else:
                runs.append(current)
                current = [point]
        runs.append(current)
        # 同长度时优先选择延伸到更靠近车头的位置。
        return max(runs, key=lambda run: (len(run), run[-1][1]))

    @staticmethod
    def _branch_candidates(corner, width):
        """Expose stable relative branch identities to the policy layer.

        The target x values are image-space association anchors, not steering
        commands. They let a roadblock box be associated with a branch while
        left/right/straight remain relative to the incoming stem.
        """
        point = corner.get('corner_point')
        stem_x = (float(corner.get('junction_stem_x') or 0.0)
                  if point is None else float(point[0]))
        if point is None:
            stem_x = width * 0.5
            target_y = 0.0
        else:
            target_y = float(point[1])
        extent = max(24.0, float(corner.get('corner_span') or 0.0) * 0.40)
        result = []
        if corner.get('junction_left'):
            result.append({'direction': 'left',
                           'target_x': max(0.0, stem_x - extent),
                           'target_y': target_y})
        if corner.get('junction_straight'):
            result.append({'direction': 'straight',
                           'target_x': stem_x, 'target_y': target_y})
        if corner.get('junction_right'):
            result.append({'direction': 'right',
                           'target_x': min(float(width - 1), stem_x + extent),
                           'target_y': target_y})
        return result

    def _detect_split_branches(self, binary, roi_top, width,
                               main_fit, main_points):
        """Build simultaneous left/straight/right paths from a shared stem.

        Dense scan rows expose secondary tape segments that diverge from the
        incoming path. Each side path must fit together with the two nearest
        stem points, which rejects disconnected chair legs and shoes.
        """
        roi_h = binary.shape[0]
        side_points = {-1: [], 1: []}
        separation = max(18.0, width * 0.055)
        first_row = 0
        last_row = int(roi_h * 0.89)
        for rel_y in range(first_row, last_row):
            xs = np.flatnonzero(binary[rel_y])
            if xs.size == 0:
                continue
            breaks = np.flatnonzero(np.diff(xs) > 1)
            groups = np.split(xs, breaks + 1)
            full_y = rel_y + roi_top
            main_x = float(np.polyval(main_fit, full_y))
            minimum = max(self.min_seg_width,
                          int(round(3 + 5 * rel_y / max(1, roi_h-1))))
            for group in groups:
                if group.size < minimum:
                    continue
                x = float(np.mean(group))
                delta = x - main_x
                if abs(delta) >= separation:
                    side_points[1 if delta > 0 else -1].append(
                        (x, float(full_y), int(group.size)))

        ordered_main = sorted(main_points, key=lambda point: point[1])
        if len(ordered_main) < 3:
            return []
        candidates = []
        ref_y = roi_top + roi_h - 1
        for side, raw in side_points.items():
            # Keep one dense, geometrically continuous side segment which can
            # reconnect to the sampled incoming stem. This rejects the bottom
            # lens/bumper shadow even when it is a long black segment.
            runs = []
            current = []
            for point in raw:
                if (current and
                        (point[1]-current[-1][1] > 2 or
                         abs(point[0]-current[-1][0]) > 14)):
                    runs.append(current)
                    current = []
                current.append(point)
            if current:
                runs.append(current)
            if not runs:
                continue
            viable = []
            for run in runs:
                if len(run) < 6 or run[-1][1]-run[0][1] < 5:
                    continue
                following = [point for point in ordered_main
                             if point[1] >= run[-1][1]]
                if not following:
                    continue
                junction = min(following, key=lambda point: point[1])
                y_gap = float(junction[1])-float(run[-1][1])
                x_gap = abs(float(junction[0])-float(run[-1][0]))
                if (y_gap <= roi_h*0.22 and
                        x_gap <= max(45.0, self.track_half*0.90)):
                    viable.append((run, junction))
            if not viable:
                continue
            run, junction = max(viable,
                                key=lambda item: (len(item[0]),
                                                  -item[1][1]))
            stride = max(1, int(math.ceil(len(run)/8.0)))
            branch_points = run[::stride]
            if branch_points[-1] != run[-1]:
                branch_points.append(run[-1])
            common = [point for point in ordered_main
                      if point[1] >= junction[1]]
            path_points = sorted(branch_points + common,
                                 key=lambda point: point[1])
            near = ordered_main[-1]
            look_x = float(np.median([point[0] for point in run]))
            look_y = float(np.median([point[1] for point in run]))
            forward = max(1.0, float(junction[1])-look_y)
            angle = math.degrees(math.atan2(
                look_x-float(junction[0]), forward))
            candidates.append({
                'direction': 'right' if side > 0 else 'left',
                'target_x': float(np.median([p[0] for p in run])),
                'target_y': float(np.median([p[1] for p in run])),
                'split_y': float(junction[1]),
                'fit_coeffs': None,
                'points': path_points,
                # Drive toward the selected side route, not the shared
                # straight stem at the bottom of the image.
                'error_px': look_x-width/2.0,
                'angle_deg': float(angle),
            })
        if not candidates:
            return []
        q2, q1, q0 = main_fit
        main_cx = float(np.clip(np.polyval(main_fit, ref_y), 0, width-1))
        straight = {
            'direction': 'straight',
            'target_x': float(np.polyval(main_fit, roi_top)),
            'target_y': float(roi_top),
            'split_y': float(max(item['split_y'] for item in candidates)),
            'fit_coeffs': tuple(float(value) for value in main_fit),
            'points': list(main_points),
            'error_px': main_cx - width/2.0,
            'angle_deg': float(math.degrees(
                math.atan(-(2*q2*ref_y + q1)))),
        }
        order = {'left': 0, 'straight': 1, 'right': 2}
        return sorted(candidates + [straight],
                      key=lambda item: order[item['direction']])

    @staticmethod
    def _robust_quadratic_fit(points, residual_limit=8.0):
        """穷举三点模型，以内点数量选二次曲线并重新拟合。"""
        if len(points) < 3:
            return None, []
        ys = np.asarray([point[1] for point in points], dtype=np.float64)
        xs = np.asarray([point[0] for point in points], dtype=np.float64)
        best_indices = np.arange(len(points))
        best_score = (-1, float('-inf'))
        for sample in combinations(range(len(points)), 3):
            sample_idx = np.asarray(sample)
            try:
                coeffs = np.polyfit(ys[sample_idx], xs[sample_idx], 2)
            except (ValueError, np.linalg.LinAlgError):
                continue
            residuals = np.abs(xs - np.polyval(coeffs, ys))
            inliers = np.flatnonzero(residuals <= residual_limit)
            if inliers.size < 3:
                continue
            score = (int(inliers.size), -float(np.mean(residuals[inliers])))
            if score > best_score:
                best_score = score
                best_indices = inliers
        try:
            coeffs = np.polyfit(ys[best_indices], xs[best_indices], 2)
        except (ValueError, np.linalg.LinAlgError):
            return None, []
        inlier_points = [points[int(index)] for index in best_indices]
        return tuple(float(value) for value in coeffs), inlier_points

    # ------------------------------------------------------------------
    def _scan_lines(self, binary, roi_top, ww, inside, pred,
                    preference_override=None):
        """对每一扫描行，在"裁切窗 ∩ 预测窗"内找最宽暗色段，返回 [(x, y, w), ...]。

        y 为整图坐标；pred 为 None 时(重捕获)只用裁切窗。
        """
        # TODO-B3【扫描行与最宽连续线段】
        # 框架已给出，学生只补下面的“最长连续非零段”搜索。
        # 每行步骤：
        #   1) 由 inside 找裁切窗左右边界 [l0, r0)；
        #   2) pred 非空时再与 [pred-track_half, pred+track_half] 求交；
        #   3) 在 seg 中从左到右扫描，把每段连续 seg[i] > 0 的
        #      起点 s 和终点 e 记录下来，保留宽度 e-s 最大的一段；
        #   4) 宽度达到 min_seg_width 才加入 points，中心为
        #      l0 + (best_s + best_e)//2；y 必须换回整幅工作图坐标。
        # 注意 best_e 是开区间终点；全零行必须跳过，不能制造中心点。
        # 验收：单行多段时选最宽段；pred 窗能排除远处干扰；重捕获可找回线。
        roi_h = binary.shape[0]
        rows = np.linspace(int(roi_h * self.scan_start_ratio),
                           roi_h - 1, self.n_scan_rows).astype(int)
        preference = (self.path_preference if preference_override is None
                      else preference_override)
        guided = preference in ('left', 'right', 'continuation')
        if guided:
            rows = rows[::-1]  # trace the connected approach from near to far
        anchor = ww/2 if pred is None else pred
        points = []
        for rel_y in rows:
            mask_row = np.nonzero(inside[rel_y])[0]
            if mask_row.size == 0:
                continue
            l0, r0 = int(mask_row[0]), int(mask_row[-1]) + 1
            if pred is not None:
                l0 = max(l0, int(round(pred - self.track_half)))
                r0 = min(r0, int(round(pred + self.track_half)) + 1)
            if r0 - l0 < 1:
                continue

            seg = binary[rel_y, l0:r0]
            best_s, best_e = -1, -1
            segments = []

            # 在此补全最长连续非零段搜索。
            # 可使用 while 循环，也可先用 np.flatnonzero 获得前景下标，
            # 再按相邻下标是否连续进行分段；不得直接取整行所有前景均值。
            i = 0
            while i < seg.size:
                if seg[i] == 0:
                    i += 1
                    continue
                start = i
                while i < seg.size and seg[i] != 0:
                    i += 1
                end = i
                segments.append((start, end))
                if best_s < 0 or end - start > best_e - best_s:
                    best_s, best_e = start, end

            minimum = max(self.min_seg_width,
                          round(3 + 5 * rel_y / max(1, roi_h-1)))
            candidates = [s for s in segments if s[1]-s[0] >= minimum]
            if candidates:
                if guided:
                    # Branch policy explicitly controls continuation choice.
                    best_s, best_e = min(
                        candidates,
                        key=lambda s: abs(l0 + (s[0]+s[1])/2-anchor))
                elif pred is not None:
                    # Hybrid mode: teacher-style multi-run scan with the
                    # current track as a soft anchor. Nearby runs win; width
                    # only breaks ties instead of dominating the choice.
                    best_s, best_e = max(
                        candidates,
                        key=lambda s: (
                            -abs(l0 + (s[0]+s[1])/2-anchor),
                            s[1]-s[0]))

            if best_s < 0:
                continue
            bw = best_e - best_s
            # 利用透视关系过滤细电线：远处道路允许较窄，越靠近车头
            # 对线宽要求越高。工作图宽度默认为 320，此时门槛约为 3~8 px。
            y_ratio = rel_y / max(1, roi_h - 1)
            perspective_min_width = int(round(3 + 5 * y_ratio))
            if bw < max(self.min_seg_width, perspective_min_width):
                continue
            cx = l0 + (best_s + best_e) // 2
            if guided:
                if bw > max(25, perspective_min_width*4):
                    if preference == 'left':
                        cx = l0 + best_s + perspective_min_width
                    elif preference == 'right':
                        cx = l0 + best_e - 1 - perspective_min_width
                    else:
                        cx = int(np.clip(anchor, l0+best_s, l0+best_e-1))
                anchor = pred = cx
            points.append((cx, rel_y + roi_top, bw))
        return sorted(points, key=lambda p: p[1]) if guided else points

    def _detect_l_corner(self, binary, roi_top):
        """检测单侧横臂的 L 弯；方向 -1=左，+1=右，0=未检测到。"""
        roi_h, ww = binary.shape[:2]
        empty = {
            'corner_dir': 0,
            'corner_point': None,
            'corner_y_ratio': 0.0,
            'corner_span': 0.0,
            'junction_left': False,
            'junction_right': False,
            'junction_near': False,
            'junction_straight': False,
            'junction_stem_x': 0.0,
            'junction_normal_width': 0.0,
            'branch_candidates': [],
            'selected_branch_direction': None,
            'line_end_candidate': False,
        }
        rows = []
        for y in range(roi_h):
            xs = np.flatnonzero(binary[y])
            if xs.size < self.min_seg_width:
                continue

            # 只使用本行最长的连续前景段。直接用 xs[0]~xs[-1] 会把墙脚、
            # 阴影等互不相连的黑块合并成一条很长的“横臂”，造成假 L 弯。
            breaks = np.flatnonzero(np.diff(xs) > 1)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks + 1, xs.size]
            lengths = ends - starts
            best = int(np.argmax(lengths))
            seg_left = int(xs[starts[best]])
            seg_right = int(xs[ends[best] - 1])
            seg_width = int(lengths[best])
            if seg_width >= self.min_seg_width:
                rows.append((y, seg_left, seg_right, seg_width))
        if len(rows) < 6:
            return empty

        # 横臂所在行的左右跨度会显著大于纵向胶带的正常宽度。
        candidate = max(rows, key=lambda item: item[2] - item[1] + 1)
        arm_y, arm_left, arm_right, _ = candidate
        span = arm_right - arm_left + 1
        lower = [item for item in rows
                 if item[0] >= arm_y + max(3, int(round(roi_h * 0.04)))]
        if len(lower) < 4:
            return empty
        normal_width = float(np.median([item[3] for item in lower]))
        # 普通圆弧在切线接近水平时也会产生一行较宽的黑带。
        # 真 L 的横臂相对线宽应该明显更长，先用长宽比排除这种
        # “渐进变宽”的弯道。
        if span < max(ww * 0.18, normal_width * 3.8):
            return empty

        # 找到横臂向下收窄成正常线宽后的进入段。真 L 在这里应该是
        # 一段近似直线；圆弧的中心会持续横向滑动，不能当成 L 主干。
        narrow_rows = [item for item in rows
                       if item[0] > arm_y and
                       item[3] <= normal_width * 1.6]
        if not narrow_rows:
            return empty
        stem_start_y = narrow_rows[0][0]
        stem_end_y = stem_start_y + max(10, int(round(roi_h * 0.20)))
        stem_rows = [item for item in rows
                     if stem_start_y <= item[0] <= stem_end_y and
                     item[3] <= normal_width * 1.8]
        if len(stem_rows) < 6:
            return empty
        stem_ys = np.asarray([item[0] for item in stem_rows], dtype=np.float64)
        stem_centers = np.asarray([(item[1] + item[2]) * 0.5
                                   for item in stem_rows], dtype=np.float64)

        # 反光会在黑胶带中间切出白缝，“最长黑段”的中心会在左右
        # 半边之间跳动。用小规模 RANSAC 找多数一致的主干，不让少数
        # 裂线点把整个真 L 否决掉。
        inlier_tol = max(2.0, normal_width * 0.25)
        best_inliers = None
        best_error = float('inf')
        for i, j in combinations(range(len(stem_rows)), 2):
            dy = stem_ys[j] - stem_ys[i]
            if abs(dy) < 1e-6:
                continue
            slope = (stem_centers[j] - stem_centers[i]) / dy
            intercept = stem_centers[i] - slope * stem_ys[i]
            residual = np.abs(stem_centers -
                              (slope * stem_ys + intercept))
            inliers = residual <= inlier_tol
            count = int(np.count_nonzero(inliers))
            error = float(np.mean(residual[inliers])) if count else float('inf')
            if (best_inliers is None or
                    count > int(np.count_nonzero(best_inliers)) or
                    (count == int(np.count_nonzero(best_inliers)) and
                     error < best_error)):
                best_inliers = inliers
                best_error = error
        required_stem_inliers = max(6, int(math.ceil(len(stem_rows) * 0.50)))
        if (best_inliers is None or
                int(np.count_nonzero(best_inliers)) < required_stem_inliers):
            return empty
        stem_ys_fit = stem_ys[best_inliers]
        stem_centers_fit = stem_centers[best_inliers]
        stem_slope, stem_intercept = np.polyfit(
            stem_ys_fit, stem_centers_fit, 1)
        stem_fit = stem_slope * stem_ys_fit + stem_intercept
        stem_rms = float(np.sqrt(np.mean((stem_centers_fit - stem_fit) ** 2)))
        # 车身不一定与进入段完全对齐，允许主干在画面中有一定
        # 斜度；普通弯道仍由前面更稳定的横臂长宽比条件排除。
        if (abs(float(stem_slope)) > 1.10 or stem_rms > inlier_tol):
            return empty
        stem_rows = [item for item, keep in zip(stem_rows, best_inliers)
                     if keep]

        # 用已经收窄且通过直线性验证的主干确定拐点，避免把
        # 横臂与主干的过渡区当成道路中心。
        stem_x = float(np.median([(item[1] + item[2]) * 0.5
                                  for item in stem_rows]))

        # 真 L 在横臂之后不会继续保持原来的纵向主干；交叉路口则有一条
        # 与下方主干对齐的线穿过横臂。反光可能抹掉一侧横臂，因此这里
        # 直接检查“上方是否仍有贯穿主干”，不依赖左右横臂是否对称。
        upper_gap = max(6, int(round(roi_h * 0.05)))
        required_upper = max(5, int(round(roi_h * 0.08)))
        align_tol = max(8.0, normal_width * 1.2)
        upper_stem_rows = [item for item in rows
                           if item[0] <= arm_y - upper_gap and
                           item[1] - align_tol <= stem_x <= item[2] + align_tol]
        upper_continues = (len(upper_stem_rows) >= required_upper and
                           upper_stem_rows[-1][0] - upper_stem_rows[0][0]
                           >= required_upper - 1)
        junction_geometry = {
            'junction_left': bool(stem_x-arm_left >= ww*0.10),
            'junction_right': bool(arm_right-stem_x >= ww*0.10),
            'junction_near': bool(arm_y/max(1, roi_h-1) >= 0.60),
        }
        if upper_continues:
            result = dict(empty)
            result.update({
                **junction_geometry,
                'corner_point': (stem_x, float(arm_y + roi_top)),
                'corner_y_ratio': float(arm_y / max(1, roi_h - 1)),
                'corner_span': float(span),
                'junction_straight': True,
                'junction_stem_x': stem_x,
                'junction_normal_width': normal_width,
            })
            return result

        left_extent = stem_x - arm_left
        right_extent = arm_right - stem_x
        margin = max(8.0, normal_width * 0.6)
        min_arm = ww * 0.10

        # 十字/T 路口即使受透视或 ROI 裁切影响，两侧横臂也可能明显不等长。
        # 只要主干左右都存在足够长的横臂，就优先归为交叉口，绝不能按
        # “较长的一边”冒充 L 弯。
        if left_extent >= min_arm and right_extent >= min_arm:
            return {**empty, **junction_geometry,
                    'corner_point': (stem_x, float(arm_y+roi_top)),
                    'corner_y_ratio': float(arm_y/max(1, roi_h-1))}
        if right_extent >= min_arm and right_extent >= left_extent + margin:
            direction = 1
        elif left_extent >= min_arm and left_extent >= right_extent + margin:
            direction = -1
        else:
            return empty                 # T/十字路口，不冒充 L 弯

        return {
            **junction_geometry,
            'corner_dir': direction,
            'corner_point': (stem_x, float(arm_y + roi_top)),
            'corner_y_ratio': float(arm_y / max(1, roi_h - 1)),
            'corner_span': float(span),
            'junction_straight': False,
            'junction_stem_x': 0.0,
            'junction_normal_width': 0.0,
        }

    def _apply_global_threshold(self, blur, inside, threshold):
        """TODO-B2a：根据极性应用一个全局阈值，并保证窗外为 0。"""
        binary = np.zeros_like(blur)
        # 黑线白底：inside 且灰度小于 threshold 的像素设为 255。
        # 白线黑底：inside 且灰度大于 threshold 的像素设为 255。
        # 请在这里补全两种极性的布尔索引赋值。
        if self.polarity == 'white':
            foreground = blur > threshold
        else:
            foreground = blur < threshold
        binary[inside & foreground] = 255
        return binary

    @staticmethod
    def _otsu(arr):
        """TODO-B2b：手写 Otsu，返回使类间方差最大的灰度阈值。"""
        # 已提供初始化，学生补全 0~255 的阈值遍历。不得调用
        # cv2.threshold(...THRESH_OTSU)，否则不能验收“手写 Otsu”。
        # 建议变量：
        #   hist     256级直方图
        #   total    像素总数
        #   sum_all  所有像素灰度总和
        #   w_b/w_f  阈值两侧的像素数
        #   m_b/m_f  两类平均灰度
        #   var      w_b*w_f*(m_b-m_f)^2
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0
        sum_all = float(np.dot(np.arange(256), hist))
        sum_b, w_b = 0.0, 0.0
        best_t, max_var = 0, -1.0

        # 在此补全遍历与最大类间方差比较，并更新 best_t。
        for threshold in range(256):
            w_b += hist[threshold]
            sum_b += threshold * hist[threshold]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f
            var = w_b * w_f * (m_b - m_f) ** 2
            if var > max_var:
                max_var = var
                best_t = threshold

        return best_t

    def _adaptive_binary(self, blur, inside):
        """TODO-B2c（基础拓展）：实现局部自适应二值化。"""
        # 可用 cv2.boxFilter 计算 adaptive_block 邻域均值 local_mean，
        # 但阈值比较与极性处理须自行完成：
        #   黑线：blur < local_mean - adaptive_c
        #   白线：blur > local_mean + adaptive_c
        # 结果只能在 inside 内为 255。禁止直接调用 cv2.adaptiveThreshold。
        binary = np.zeros_like(blur)
        local_mean = cv2.boxFilter(
            blur, cv2.CV_32F,
            (self.adaptive_block, self.adaptive_block),
            normalize=True, borderType=cv2.BORDER_REPLICATE)
        if self.polarity == 'white':
            foreground = blur.astype(np.float32) > local_mean + self.adaptive_c
        else:
            foreground = blur.astype(np.float32) < local_mean - self.adaptive_c
        binary[inside & foreground] = 255
        return binary

    def _empty_result(self, binary=None, roi_top=0):
        return {
            'is_valid': False,
            'centroid': None,
            'error_px': 0.0,
            'angle_deg': 0.0,
            'fit_coeffs': None,
            'corner_dir': 0,
            'corner_point': None,
            'corner_y_ratio': 0.0,
            'corner_span': 0.0,
            'junction_left': False,
            'junction_right': False,
            'junction_near': False,
            'junction_straight': False,
            'junction_stem_x': 0.0,
            'junction_normal_width': 0.0,
            'branch_candidates': [],
            'line_end_candidate': False,
            'line_width_mm': None,
            'points': [],
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
        }


# =====================================================================
# 2. 巡线主控制器（简化 PD + 角度前馈）
# =====================================================================
class LineFollower:
    """
    巡线主控制器：相机取帧 → 检测 → PD 控制 → 串口底盘。

    参数（均为底盘串口约定单位）：
      base_speed  直道巡航速度 mm/s
      max_z       最大转向速度 mrad/s
      kp / kd / ka  横向误差比例/微分/角度前馈增益
      err_alpha   误差/角度低通滤波系数(0~1)
      lost_hold   失线低速直行的帧数上限，超过则停车
      startup_frames 起步确认帧数：连续检测到线这么多帧后车辆才开始前进(默认5)
      ramp_frames    起步后速度从0平滑加速到目标的帧数(默认20，约1秒)
      corner_delay_frames 确认L弯后低速直行多少帧再转向；越大转得越晚(默认10)
      corner_delay_speed  L弯延迟直行阶段的速度 mm/s(默认40)
      corner_turn_degrees L弯原地旋转的目标角度；越大转得越多(默认78度)
      corner_turn_speed   L弯原地旋转的目标速度 mrad/s(默认300)
      start_rotate   起步确认期间是否原地转向对准线(默认False:静止确认后边前进边修正)
    """

    def __init__(self, camera, chassis,
                 base_speed=160, max_z=800,
                 kp=12.0, kd=1.2, ka=3.5,
                 err_alpha=0.6, z_rate_limit=120.0,
                 lost_hold=10, search_frames=15,
                 startup_frames=5, ramp_frames=20,
                 corner_delay_frames=10, corner_delay_speed=40,
                 corner_turn_degrees=78.0, corner_turn_speed=300,
                 start_rotate=False,
                 work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, scan_start_ratio=0.25,
                 crop_bottom_frac=0.70, crop_top_frac=0.90,
                 track_half=60.0, polarity='black',
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0,
                 z_invert=True,   # 转向方向取反（默认 True）
                 target_fps=20, debug=False, web_debug=None,
                 line_width_model=None, enforce_width=False):
        self.camera = camera
        self.chassis = chassis
        self.base_speed = base_speed
        self.max_z = max_z
        self.kp = kp
        self.kd = kd
        self.ka = ka
        self.err_alpha = err_alpha
        self.z_rate_limit = z_rate_limit    # 每帧最大转向增量(mrad/s)，防猛甩
        self.lost_hold = lost_hold
        self.search_frames = search_frames  # 失线后低速旋转搜索的帧数
        self.z_invert = z_invert            # 转向方向取反(硬件/装向与协议约定相反时使用)
        self.startup_frames = startup_frames  # 起步确认帧数：线连续稳定这么多帧后才前进
        self.ramp_frames = ramp_frames        # 起步后速度从0平滑加速到目标所用帧数
        self.corner_delay_frames = max(0, int(corner_delay_frames))
        self.corner_delay_speed = max(0, int(corner_delay_speed))
        self.corner_turn_radians = math.radians(
            float(np.clip(corner_turn_degrees, 10.0, 180.0)))
        self.corner_turn_speed = int(np.clip(corner_turn_speed, 50, 1000))
        self.start_rotate = start_rotate      # 起步是否原地转向对准线(默认关，静止确认后前进)

        self.detector = LineDetector(
            work_width=work_width, roi_top_ratio=roi_top_ratio,
            n_scan_rows=n_scan_rows, scan_start_ratio=scan_start_ratio,
            crop_bottom_frac=crop_bottom_frac, crop_top_frac=crop_top_frac,
            track_half=track_half, polarity=polarity,
            binary_mode=binary_mode, fixed_threshold=fixed_threshold,
            adaptive_block=adaptive_block, adaptive_c=adaptive_c,
            line_width_model=line_width_model, enforce_width=enforce_width)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1, target_fps)
        self.debug = debug
        self.web_debug = web_debug

        self._prev_err = 0.0
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._has_prev = False      # 是否已有上一帧有效误差(首帧不微分)
        self._last_z = 0.0
        self._lost_entry_z = 0.0   # 失线瞬间的原始转向（反相前）
        self._last_sign = 1         # 最后一次有效误差方向(失线搜索用)
        self._lost_count = 0
        self._no_frame_count = 0
        self._start_seen = 0      # 起步期连续有效帧计数
        self._started = False     # 起步确认是否完成(完成后才前进)
        self._run_frames = 0      # 起步后已运行帧数(速度斜坡用)
        self._corner_dir = 0      # 正在执行的 L 弯方向：-1左，+1右
        self._corner_frames = 0
        self._corner_phase = ''   # advance: 越过拐点；turn: 原地转向
        self._corner_turn_radians = 0.0
        self._corner_confirm_dir = 0
        self._corner_confirm_count = 0
        self._corner_exit_frames = 0  # 完成一个 L 后短暂忽略旧拐角，允许连续 L
        self.fps = 0.0
        self._fps_n = 0
        self._fps_t = time.time()
        self.odometry = ImuOdometry()
        self._last_chassis_status = None
        self._manual_mode = threading.Event()
        self._resume_tracking = threading.Event()
        self._manual_lock = threading.Lock()
        self._manual_target = None
        self._manual_state = '视觉循迹'
        self._manual_phase = 'idle'
        self._manual_remaining_m = 0.0
        self._manual_remaining_deg = 0.0

    def reset_odometry(self):
        self.odometry.reset()
        logger.info('IMU 里程计已清零')

    @staticmethod
    def _angle_error(target_deg, current_deg):
        return (target_deg - current_deg + 180.0) % 360.0 - 180.0

    def set_manual_mode(self, enabled):
        enabled = bool(enabled)
        if enabled:
            self._manual_mode.set()
            with self._manual_lock:
                self._manual_target = None
                self._manual_state = '手动模式待命'
                self._manual_phase = 'idle'
                self._manual_remaining_m = 0.0
                self._manual_remaining_deg = 0.0
            self.chassis.stop()
            logger.warning('已切换到手动里程控制，视觉控制暂停')
        else:
            self.chassis.stop()
            with self._manual_lock:
                self._manual_target = None
                self._manual_state = '视觉循迹'
                self._manual_phase = 'idle'
                self._manual_remaining_m = 0.0
                self._manual_remaining_deg = 0.0
            self._resume_tracking.set()
            self._manual_mode.clear()
            logger.warning('已退出手动里程控制，恢复视觉循迹')
        return self.manual_control_status()

    def start_manual_target(self, distance_m, angle_deg):
        if not self._manual_mode.is_set():
            raise ValueError('请先打开手动里程控制')
        distance_m = float(distance_m)
        angle_deg = float(angle_deg)
        if (not math.isfinite(distance_m) or
                not -5.0 <= distance_m <= 5.0):
            raise ValueError('距离必须在 -5.0~5.0 m 之间')
        if (not math.isfinite(angle_deg) or
                not -360.0 <= angle_deg <= 360.0):
            raise ValueError('角度必须在 -360~360° 之间')

        pose = self.odometry.snapshot()
        target_yaw_total = pose['odom_yaw_total_deg'] + angle_deg
        rotate_seconds = abs(angle_deg) / 17.0
        drive_seconds = abs(distance_m) / 0.12
        with self._manual_lock:
            self._manual_target = {
                'distance_m': distance_m,
                'angle_deg': angle_deg,
                'target_yaw_total_deg': target_yaw_total,
                'drive_yaw_deg': pose['odom_yaw_deg'],
                'drive_start_x_m': pose['odom_x_m'],
                'drive_start_y_m': pose['odom_y_m'],
                'deadline': time.monotonic() +
                            3.0 + 2.5 * (rotate_seconds + drive_seconds),
            }
            self._manual_phase = ('rotate' if abs(angle_deg) > 2.0
                                  else 'drive')
            self._manual_state = ('正在转向' if self._manual_phase == 'rotate'
                                  else '正在行驶')
            self._manual_remaining_m = abs(distance_m)
            self._manual_remaining_deg = abs(angle_deg)
        logger.info('手动里程目标: 距离=%+.3fm 相对角度=%+.1f°',
                    distance_m, angle_deg)
        return self.manual_control_status()

    def cancel_manual_target(self):
        with self._manual_lock:
            self._manual_target = None
            self._manual_phase = 'idle'
            self._manual_state = ('手动模式待命' if self._manual_mode.is_set()
                                  else '视觉循迹')
            self._manual_remaining_m = 0.0
            self._manual_remaining_deg = 0.0
        self.chassis.stop()
        logger.warning('手动里程目标已取消并停车')
        return self.manual_control_status()

    def manual_control_status(self):
        with self._manual_lock:
            target = dict(self._manual_target or {})
            return {
                'manual_mode': self._manual_mode.is_set(),
                'manual_state': self._manual_state,
                'manual_phase': self._manual_phase,
                'manual_target_distance_m': target.get('distance_m', 0.0),
                'manual_target_angle_deg': target.get('angle_deg', 0.0),
                'manual_remaining_m': self._manual_remaining_m,
                'manual_remaining_deg': self._manual_remaining_deg,
            }

    def _reset_tracking_state(self):
        self._prev_err = 0.0
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._has_prev = False
        self._last_z = 0.0
        self._lost_entry_z = 0.0
        self._lost_count = 0
        self._start_seen = 0
        self._started = False
        self._run_frames = 0
        self._corner_dir = 0
        self._corner_frames = 0
        self._corner_phase = ''
        self._corner_turn_radians = 0.0
        self._corner_exit_frames = 0

    def _manual_control_step(self):
        pose = self.odometry.snapshot()
        now = time.monotonic()
        with self._manual_lock:
            target = self._manual_target
            if target is None:
                return 'manual-idle', 0, 0
            if now >= target['deadline']:
                self._manual_target = None
                self._manual_phase = 'idle'
                self._manual_state = '目标超时，已停车'
                self._manual_remaining_m = 0.0
                self._manual_remaining_deg = 0.0
                self.chassis.stop()
                return 'manual-timeout', 0, 0

            yaw_error = (target['target_yaw_total_deg'] -
                         pose['odom_yaw_total_deg'])
            self._manual_remaining_deg = abs(yaw_error)
            if self._manual_phase == 'rotate':
                # 根据当前实测角速度预留约 120ms 的制动角，防止底盘
                # 在发出零速后仍因惯性继续转动。
                real_z = 0.0
                if self._last_chassis_status is not None:
                    real_z = abs(float(
                        self._last_chassis_status.get('real_z', 0.0)))
                braking_deg = math.degrees(real_z * 0.12)
                stop_tolerance = max(1.0, min(3.0, braking_deg))
                if abs(yaw_error) <= stop_tolerance:
                    self.chassis.send_speed(0, 0, 0)
                    if abs(target['distance_m']) <= 0.01:
                        self._manual_target = None
                        self._manual_phase = 'idle'
                        self._manual_state = '目标完成，已停车'
                        self._manual_remaining_m = 0.0
                        self._manual_remaining_deg = 0.0
                        self.chassis.stop()
                        return 'manual-complete', 0, 0
                    # 先等待车体停稳，否则旋转惯性会带着直线阶段偏离。
                    target['settle_until'] = now + 0.25
                    self._manual_phase = 'settle'
                    self._manual_state = '转向完成，等待停稳'
                    return 'manual-transition', 0, 0
                # 距离目标越近转得越慢；降低最小转速可显著减小越界。
                turn_mag = min(240.0, max(55.0, abs(yaw_error) * 7.0))
                # 世界航向左正；底盘协议 z 右正，符号相反。
                z_speed = -int(math.copysign(turn_mag, yaw_error))
                self._manual_state = '正在转向'
                self.chassis.send_speed(0, 0, z_speed)
                return 'manual-rotate', 0, z_speed

            if self._manual_phase == 'settle':
                self.chassis.send_speed(0, 0, 0)
                if now < target['settle_until']:
                    return 'manual-settle', 0, 0
                target['drive_yaw_deg'] = pose['odom_yaw_deg']
                target['drive_start_x_m'] = pose['odom_x_m']
                target['drive_start_y_m'] = pose['odom_y_m']
                self._manual_phase = 'drive'
                self._manual_state = '正在行驶'
                yaw_error = 0.0

            # 用“起点到当前位置在目标航向上的投影”计算进度。
            # 这样原地转动、横向滑动不会被误算成前进距离。
            heading = math.radians(target['drive_yaw_deg'])
            dx = pose['odom_x_m'] - target['drive_start_x_m']
            dy = pose['odom_y_m'] - target['drive_start_y_m']
            along = dx * math.cos(heading) + dy * math.sin(heading)
            direction = 1.0 if target['distance_m'] >= 0.0 else -1.0
            travelled = max(0.0, direction * along)
            remaining = max(0.0, abs(target['distance_m']) - travelled)
            self._manual_remaining_m = remaining
            if remaining <= 0.008:
                self._manual_target = None
                self._manual_phase = 'idle'
                self._manual_state = '目标完成，已停车'
                self._manual_remaining_m = 0.0
                self._manual_remaining_deg = 0.0
                self.chassis.stop()
                return 'manual-complete', 0, 0

            # 在最后 24cm 内按剩余距离连续降速，减少停车越界。
            linear_mag = min(120.0, max(30.0, remaining * 500.0))
            x_speed = int(math.copysign(linear_mag, target['distance_m']))
            drive_yaw_error = self._angle_error(
                target['drive_yaw_deg'], pose['odom_yaw_deg'])
            z_speed = int(np.clip(-drive_yaw_error * 8.0, -180.0, 180.0))
            self._manual_state = '正在行驶'
            self.chassis.send_speed(x_speed, 0, z_speed)
            return 'manual-drive', x_speed, z_speed

    # ------------------------------------------------------------------
    def run(self, max_frames=None, stop_event=None):
        """主循环。"""
        logger.info('巡线启动: 极性=%s base=%dmm/s max_z=%dmrad/s',
                    self.detector.polarity, self.base_speed, self.max_z)
        frame_count = 0
        last_t = time.time()
        start_time = last_t
        state = 'run'
        status_t = 0.0
        send_fail = 0

        try:
            while not (stop_event and stop_event.is_set()):
                frame = self.camera.read()
                if stop_event and stop_event.is_set():
                    logger.warning('收到网页急停，退出控制循环')
                    break
                now = time.time()
                dt = max(now - last_t, 1e-3)
                last_t = now

                if frame is None:
                    self._no_frame_count += 1
                    if self._no_frame_count >= 60:   # 约 3 秒无帧，停车退出
                        logger.error('连续 %d 帧无图像，相机可能掉线，停车退出',
                                     self._no_frame_count)
                        self.chassis.stop()
                        break
                else:
                    self._no_frame_count = 0

                det = self.detector.process(frame)
                err = det['error_px']
                angle = det['angle_deg']
                p_term = d_term = angle_term = 0.0

                if self._manual_mode.is_set():
                    chassis_status = self.chassis.read_status()
                    if chassis_status is not None:
                        self._last_chassis_status = chassis_status
                        self.odometry.update(chassis_status)
                    state, speed, z = self._manual_control_step()
                    odometry = self.odometry.snapshot()
                    if self.web_debug is not None:
                        web_det = dict(det)
                        web_det['work_width'] = self.detector.work_width
                        web_det['crop_top_frac'] = self.detector.crop_top_frac
                        web_det['crop_bottom_frac'] = self.detector.crop_bottom_frac
                        self.web_debug.update(frame, web_det, {
                            'state': state,
                            'frame_count': frame_count,
                            'error_px': float(err),
                            'angle_deg': float(angle),
                            'speed': int(speed),
                            'turn': float(z),
                            'p_term': 0.0,
                            'd_term': 0.0,
                            'angle_term': 0.0,
                            'fps': float(self.fps),
                            'max_z': abs(float(self.max_z)),
                            'lost_count': self._lost_count,
                            'no_frame_count': self._no_frame_count,
                            'start_seen': self._start_seen,
                            'startup_frames': self.startup_frames,
                            'started': self._started,
                            'binary_mode': self.detector.binary_mode,
                            **odometry,
                            **self.manual_control_status(),
                        })
                        if self.web_debug.restart_requested:
                            logger.info('收到网页参数更新，停车后重启')
                            break
                    if self.debug:
                        self._show_debug(frame, det, state, speed, z)
                    elapsed = time.time() - now
                    if elapsed < self.frame_interval:
                        time.sleep(self.frame_interval - elapsed)
                    self._fps_n += 1
                    if time.time() - self._fps_t >= 1.0:
                        self.fps = self._fps_n / (time.time() - self._fps_t)
                        self._fps_n = 0
                        self._fps_t = time.time()
                    frame_count += 1
                    if max_frames is not None and frame_count >= max_frames:
                        break
                    continue

                if self._resume_tracking.is_set():
                    self._reset_tracking_state()
                    self._resume_tracking.clear()

                detected_corner = int(det.get('corner_dir', 0))
                if self._corner_exit_frames > 0:
                    self._corner_exit_frames -= 1
                    observed_corner = 0
                else:
                    observed_corner = detected_corner
                corner_near = float(det.get('corner_y_ratio', 0.0)) >= 0.52
                if (self._corner_dir == 0 and self._started and
                        det['is_valid'] and observed_corner and corner_near):
                    self._corner_dir = observed_corner
                    self._corner_frames = 0
                    self._corner_phase = 'advance'
                    self._corner_turn_radians = 0.0
                    logger.info('识别到%s L 弯，跨度=%.0fpx，立即进入后续流程',
                                '左' if observed_corner < 0 else '右',
                                float(det.get('corner_span', 0.0)))
                elif self._corner_dir == 0:
                    self._corner_confirm_dir = 0
                    self._corner_confirm_count = 0

                corner_handled = False
                if self._corner_dir:
                    # 识别到 L 后不立刻转：按可调帧数低速直行，让车身中心
                    # 到达拐点后再进入有界的原地转向。
                    if self._corner_phase == 'advance':
                        if self._corner_frames >= self.corner_delay_frames:
                            self._corner_phase = 'turn'
                            self._corner_frames = 0
                            logger.info('%s L 弯已到近处，开始受限原地转向',
                                        '左' if self._corner_dir < 0 else '右')
                        else:
                            self._corner_frames += 1
                            z = 0
                            speed = min(self.corner_delay_speed,
                                        max(0, int(self.base_speed)))
                            state = ('corner-delay-left' if self._corner_dir < 0
                                     else 'corner-delay-right')
                            if self.base_speed <= 0:
                                speed = 0
                            if self.chassis.send_speed(speed, 0, 0):
                                send_fail = 0
                            else:
                                send_fail += 1
                            corner_handled = True

                    # 出口线转成近似纵向后即可结束，不再强制长时间旋转。
                    reacquired = (self._corner_phase == 'turn' and
                                  self._corner_turn_radians >= 0.75 and
                                  det['is_valid'] and
                                  observed_corner == 0 and abs(angle) < 35 and
                                  abs(err) < 55)
                    if not corner_handled and reacquired:
                        logger.info('%s L 弯出口已重新捕获',
                                    '左' if self._corner_dir < 0 else '右')
                        self._corner_dir = 0
                        self._corner_frames = 0
                        self._corner_phase = ''
                        self._corner_turn_radians = 0.0
                        self._corner_confirm_dir = 0
                        self._corner_confirm_count = 0
                        self._corner_exit_frames = 15
                        self._has_prev = False
                        self._last_z = 0.0
                    elif not corner_handled:
                        self._corner_frames += 1
                        self._lost_count = 0
                        self._has_prev = False
                        self._filtered_err = 0.0
                        self._filtered_angle = 0.0
                        turn_limit = min(abs(float(self.max_z)),
                                         float(self.corner_turn_speed))
                        turn_mag = min(turn_limit,
                                       100.0 + self._corner_frames * 15.0)
                        raw_z = self._corner_dir * turn_mag
                        self._last_sign = self._corner_dir
                        turn_complete = (self._corner_turn_radians >= self.corner_turn_radians or
                                         self._corner_frames > 110)
                        if turn_complete:
                            logger.info('%s L 弯旋转完成 %.1f°，进入低速循迹退出阶段',
                                        '左' if self._corner_dir < 0 else '右',
                                        math.degrees(self._corner_turn_radians))
                            self._corner_dir = 0
                            self._corner_frames = 0
                            self._corner_phase = ''
                            self._corner_turn_radians = 0.0
                            self._corner_confirm_dir = 0
                            self._corner_confirm_count = 0
                            self._corner_exit_frames = 15
                            self._has_prev = False
                            self._last_z = 0.0
                            z = 0
                            speed = 0
                            state = 'corner-exit'
                            if self.chassis.send_speed(0, 0, 0):
                                send_fail = 0
                            else:
                                send_fail += 1
                            corner_handled = True
                        else:
                            self._corner_turn_radians += abs(raw_z) * dt / 1000.0
                            self._last_z = raw_z
                            z = -raw_z if self.z_invert else raw_z
                            if self.base_speed <= 0:
                                z = 0
                            speed = 0
                            state = ('corner-left' if self._corner_dir < 0
                                     else 'corner-right')
                            if self.chassis.send_speed(0, 0, int(z)):
                                send_fail = 0
                            else:
                                send_fail += 1
                            corner_handled = True

                if corner_handled:
                    pass
                elif det['is_valid']:
                    state = 'run'
                    self._lost_count = 0
                    self._last_sign = 1 if err >= 0 else -1

                    if not self._has_prev:
                        # 首帧：用真实误差初始化滤波器，避免 derr 尖峰把转向打满
                        self._filtered_err = err
                        self._filtered_angle = angle
                        self._prev_err = err
                        self._has_prev = True

                    # 低通滤波（对误差与角度统一滤波）
                    self._filtered_err = (self.err_alpha * err +
                                          (1 - self.err_alpha) * self._filtered_err)
                    self._filtered_angle = (self.err_alpha * angle +
                                            (1 - self.err_alpha) * self._filtered_angle)
                    err = self._filtered_err
                    angle = self._filtered_angle

                    # TODO-B5【PD + 方向角前馈】
                    # 先计算误差变化率 derr，再分别计算 P、D、方向角前馈三项；
                    # 相加得到 z_raw，并把结果限制到 [-max_z, max_z]。
                    # 注意 dt 已做下限保护；首个有效帧在上方已初始化，避免微分冲击。
                    # 验收：线向右移时原始 z 符号应指向右转；阶跃误差下输出不应失控。
                    derr = (err - self._prev_err) / dt
                    p_term = self.kp * err
                    d_term = self.kd * derr
                    angle_term = self.ka * angle
                    z_raw = p_term + d_term + angle_term
                    z = float(np.clip(
                        z_raw, -abs(self.max_z), abs(self.max_z)))

                    # 转向速率限制：单帧最多变化 z_rate_limit mrad/s，防车身猛甩
                    dz = z - self._last_z
                    if abs(dz) > self.z_rate_limit:
                        z = self._last_z + self.z_rate_limit * (1 if dz > 0 else -1)
                    self._last_z = z

                    if self.z_invert:          # 转向方向取反（z>0 左转 / z<0 右转）
                        z = -z
                    if self.base_speed <= 0:
                        z = 0                  # speed=0 是真正的静止调试模式

                    # TODO-B7a【起步确认】
                    # 只有连续 startup_frames 帧检测有效，才允许 self._started=True；
                    # 任一无效帧会在下方清零 _start_seen。补全前条件恒假，车辆不会起步。
                    if not self._started:
                        self._start_seen += 1
                        startup_confirmed = self._start_seen >= max(1, self.startup_frames)
                        if startup_confirmed:
                            self._started = True      # 本帧起开始前进
                            self._run_frames = 0
                            state = 'start-ok'
                        else:
                            state = 'start'           # 静止确认：等线稳定，默认不原地转动
                            speed = 0
                            # 仅在开启 start_rotate 且线路明显偏离时才原地对准，避免起步就转头
                            cmd_z = int(z) if self.start_rotate and abs(err) > 20 else 0
                            if self.chassis.send_speed(0, 0, cmd_z):
                                send_fail = 0
                            else:
                                send_fail += 1
                            self._prev_err = err

                    # TODO-B6【起步斜坡 + 弯道降速】
                    # ramp 应在 ramp_frames 内从接近0逐步增至1；再根据 abs(z)/max_z
                    # 连续降低速度，转向越强速度越低。abs(err)>40 时把速度上限压到
                    # base_speed 的30%。注意 ramp_frames/max_z 可能为0，必须防止除零。
                    # 验收：起步速度逐帧平滑增加；直道快、急弯明显慢；始终不超base_speed。
                    if self._started:
                        self._run_frames += 1
                        if self.ramp_frames <= 0:
                            ramp = 1.0
                        else:
                            ramp = min(1.0, self._run_frames / self.ramp_frames)
                        max_turn = max(1.0, abs(float(self.max_z)))
                        turn_ratio = min(1.0, abs(z) / max_turn)
                        curve_scale = max(0.3, 1.0 - 0.7 * turn_ratio)
                        speed = int(round(self.base_speed * ramp * curve_scale))
                        if abs(err) > 40:
                            speed = min(speed, int(round(self.base_speed * 0.3)))
                        if self._corner_exit_frames > 0:
                            # 已完成的旧拐角可能仍在高位摄像头视野内。退出阶段
                            # 保留正常循迹转向，只限制前进速度，避免再次触发旧 L。
                            speed = min(speed, int(round(self.base_speed * 0.35)))
                            state = 'corner-exit'
                        elif observed_corner:
                            # 拐点尚远时继续沿主干靠近，但预先减速；达到触发线后
                            # 上面的确认逻辑会切换为原地转向。
                            speed = min(speed, int(round(self.base_speed * 0.35)))
                            z = 0
                            self._last_z = 0.0
                            state = ('corner-approach-left' if observed_corner < 0
                                     else 'corner-approach-right')
                        if self.chassis.send_speed(speed, 0, int(z)):
                            send_fail = 0
                        else:
                            send_fail += 1
                        self._prev_err = err
                else:
                    self._has_prev = False
                    self._filtered_err = 0.0
                    self._filtered_angle = 0.0
                    if not self._started:
                        # 起步期间失线：不前进也不旋转搜索，原地停车等线出现
                        self._last_z = 0.0
                        self._lost_entry_z = 0.0
                        self._start_seen = 0
                        state, z, speed = 'start-lost', 0, 0
                        self.chassis.send_speed(0, 0, 0)
                    else:
                        state, z, speed = self._handle_lost()

                chassis_status = self.chassis.read_status()
                if chassis_status is not None:
                    self._last_chassis_status = chassis_status
                    self.odometry.update(chassis_status)
                odometry = self.odometry.snapshot()

                if self.web_debug is not None:
                    web_det = dict(det)
                    web_det['work_width'] = self.detector.work_width
                    web_det['crop_top_frac'] = self.detector.crop_top_frac
                    web_det['crop_bottom_frac'] = self.detector.crop_bottom_frac
                    self.web_debug.update(frame, web_det, {
                        'state': state,
                        'frame_count': frame_count,
                        'error_px': float(err),
                        'angle_deg': float(angle),
                        'speed': int(speed),
                        'turn': float(z),
                        'p_term': float(p_term),
                        'd_term': float(d_term),
                        'angle_term': float(angle_term),
                        'fps': float(self.fps),
                        'max_z': abs(float(self.max_z)),
                        'lost_count': self._lost_count,
                        'no_frame_count': self._no_frame_count,
                        'start_seen': self._start_seen,
                        'startup_frames': self.startup_frames,
                        'started': self._started,
                        'binary_mode': self.detector.binary_mode,
                        'corner_phase': self._corner_phase,
                        'corner_exit_frames': self._corner_exit_frames,
                        'corner_delay_frames': self.corner_delay_frames,
                        'corner_delay_speed': self.corner_delay_speed,
                        'corner_turn_target_deg': math.degrees(self.corner_turn_radians),
                        'corner_turn_speed': self.corner_turn_speed,
                        'corner_turn_deg': math.degrees(self._corner_turn_radians),
                        **odometry,
                        **self.manual_control_status(),
                    })
                    if self.web_debug.restart_requested:
                        logger.info('收到网页参数更新，停车后重启')
                        break

                if self.debug:
                    self._show_debug(frame, det, state, speed, z)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info('调试窗口按 q 退出')
                        break

                # 帧率控制与统计
                elapsed = time.time() - now
                sleep_t = self.frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                self._fps_n += 1
                if time.time() - self._fps_t >= 1.0:
                    self.fps = self._fps_n / (time.time() - self._fps_t)
                    self._fps_n = 0
                    self._fps_t = time.time()

                if time.time() - status_t >= 1.0:
                    status_t = time.time()
                    if send_fail >= 10:
                        logger.error('串口发送持续失败(%d次)，停车退出', send_fail)
                        self.chassis.stop()
                        break
                    # 回读底盘状态用于显示（单位已换算）
                    real_dps = ''
                    st = self._last_chassis_status
                    if st is not None:
                        real_dps = f" 实转z={st['real_z'] * 1000:+.0f}mrad/s"
                        real_dps += f" 实x/y={st['real_x']:+d}/{st['real_y']:+d}mm/s"
                        real_dps += f" 陀螺z={st['ang_vel_z']:+d}"
                    logger.info(
                        '状态=%-8s 极性=%-5s | err=%6.1fpx 角度=%5.1f° | '
                        '速度=%3dmm/s 转向=%4dmrad/s | FPS=%.0f | 失帧=%d%s',
                        state, det['line_type'], err, angle,
                        int(speed), int(z), self.fps, self._no_frame_count, real_dps)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    logger.info('达到最大帧数 %d，退出', max_frames)
                    break

        except KeyboardInterrupt:
            logger.info('用户中断')
        finally:
            self.chassis.stop()
            cv2.destroyAllWindows()
            logger.info('巡线结束，运行 %.1fs', time.time() - start_time)

    # ------------------------------------------------------------------
    def _handle_lost(self):
        """失线处理(分三阶段，全部低速安全执行)：
        1) lost-hold   低速前进并保留失线前转向，再逐渐衰减；
        2) lost-search 朝失线前的实际转向方向旋转搜索，超过半程后自动换向，
                       避免单一方向找不回(曲线弯后线可能跑向另一侧)；
        3) lost-stop   仍未找回则停车等待。
        返回 (state, z, speed)。
        """
        # TODO-B7b【失线安全】
        # 必须实现有界的三阶段处理：
        #   lost-hold：最多 lost_hold 帧，仅允许低速短暂保持；
        #   lost-search：最多 search_frames 帧，按最后误差方向低速搜索；
        #   lost-stop：超过总帧数后发送零速并持续停车。
        # 所有非零速度都必须有明确帧数上限；不允许无限前进或无限旋转。
        # 验收：遮住线路后车辆在规定时间内停车，重新出现线路后可恢复。
        self._lost_count += 1
        if self._lost_count == 1:
            self._lost_entry_z = float(self._last_z)
        if self.base_speed <= 0:
            self._last_z = 0.0
            self.chassis.send_speed(0, 0, 0)
            return 'lost-stop', 0, 0
        hold_limit = max(0, int(self.lost_hold))
        search_limit = max(0, int(self.search_frames))

        if self._lost_count <= hold_limit:
            speed = max(0, int(round(self.base_speed * 0.25)))
            # 从失线前转向平滑衰减到 30%，短暂遮挡时继续沿原弯道
            # 走，避免立即归零后沿切线驶出路径。
            progress = self._lost_count / max(1.0, float(hold_limit))
            hold_z_raw = self._lost_entry_z * (1.0 - 0.70 * progress)
            self._last_z = hold_z_raw
            z = -hold_z_raw if self.z_invert else hold_z_raw
            z = float(np.clip(z, -abs(self.max_z), abs(self.max_z)))
            self.chassis.send_speed(speed, 0, int(z))
            return 'lost-hold', z, speed

        search_index = self._lost_count - hold_limit
        if search_index <= search_limit:
            # 横向误差的符号不一定等于转向方向（角度前馈可能占
            # 主导），优先使用失线前的真实控制转向符号。
            if abs(self._lost_entry_z) >= 20.0:
                direction = 1 if self._lost_entry_z > 0 else -1
            else:
                direction = self._last_sign
            if search_index > (search_limit + 1) // 2:
                direction = -direction
            z_raw = direction * min(abs(float(self.max_z)), 200.0)
            self._last_z = z_raw
            z = z_raw
            if self.z_invert:
                z = -z
            self.chassis.send_speed(0, 0, int(z))
            return 'lost-search', z, 0

        self._last_z = 0.0
        self._lost_entry_z = 0.0
        self.chassis.send_speed(0, 0, 0)
        return 'lost-stop', 0, 0

    # ------------------------------------------------------------------
    def _show_debug(self, frame, det, state, speed, z):
        """绘制调试窗口：原图 + 二值图 + 当前控制量。"""
        if frame is None:
            return
        disp_w = 480
        h, w = frame.shape[:2]
        disp = cv2.resize(frame, (disp_w, int(disp_w * h / w)))

        binary = det['binary']
        if binary is not None:
            bh, bw = binary.shape[:2]
            # 翻转显示：黑线直接显示为黑、其他区域为白（贴合真实画面观感）
            bdisp = cv2.bitwise_not(binary)
            bdisp = cv2.resize(bdisp, (disp_w, int(disp_w * bh / bw)))
            bdisp = cv2.cvtColor(bdisp, cv2.COLOR_GRAY2BGR)
            disp = np.vstack([disp, bdisp])

        cx0 = int(disp_w / 2)
        cv2.line(disp, (cx0, 0), (cx0, disp.shape[0]), (255, 0, 0), 1)

        if det['is_valid']:
            s = disp.shape[1] / self.detector.work_width
            for (x, y, bw) in det['points']:
                cv2.circle(disp, (int(x * s), int(y * s)), 3, (0, 255, 0), -1)
            y_top = min(p[1] for p in det['points'])
            y_bot = max(p[1] for p in det['points'])
            fit_ys = np.linspace(y_top, y_bot, 40)
            fit_xs = np.polyval(det['fit_coeffs'], fit_ys)
            curve = np.column_stack((fit_xs * s, fit_ys * s))
            curve[:, 0] = np.clip(curve[:, 0], 0, disp_w - 1)
            cv2.polylines(disp, [np.rint(curve).astype(np.int32)], False,
                          (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cx, cy = det['centroid']
            cv2.circle(disp, (int(cx * s), int(cy * s)), 6, (0, 255, 255), -1)

        info = (f"state={state} x={speed}mm/s z={z:.0f}mrad/s | "
                f"err={det['error_px']:.1f}px ang={det['angle_deg']:.1f}deg | "
                f"polar={det['line_type']} fps={self.fps:.0f}")
        cv2.putText(disp, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.imshow('LineFollower (04)', disp)
