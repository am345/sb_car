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

import cv2
import numpy as np

from core.corner_maneuver import CornerManeuver
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

        threshold = None
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
        shared_window = np.array_equal(inside, branch_inside)
        if shared_window:
            branch_binary = binary
        elif self.binary_mode == 'adaptive':
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
        if self.binary_mode != 'adaptive':
            branch_mask = self._recover_reflective_tape(
                blur, branch_mask, branch_inside, threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        binary = (branch_binary if shared_window else
                  cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))

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
            # A genuine straight tape stripe is itself a nearly solid
            # rectangle.  Fill ratio alone therefore cannot distinguish it
            # from a shadow.  Only reject a solid component here when it is
            # also implausibly broad; calibrated metric width performs the
            # final 10--100 mm guard below on the real vehicle.
            if (bw_ > ww * 0.30 and bh_ > 8 and fill > 0.85):
                continue
            # A side-attached, broad porous wedge is typically a floor shadow
            # or furniture edge, not a tape ribbon. Keep central wide blobs so
            # genuine intersections remain available to the branch detector.
            side_attached = x_ <= ww * 0.12 or x_ + bw_ >= ww * 0.88
            if (side_attached and bw_ > ww * 0.30 and
                    bh_ > roi_h * 0.60 and
                    not self._side_candidate_reconnects(binary, labels, i,
                                                        ww, roi_h)):
                continue
            line_mask[labels == i] = 255
        binary = line_mask
        if self.binary_mode != 'adaptive':
            # Recover only from components that already passed strict geometry
            # filtering. Weak scratches cannot promote themselves into roads.
            binary = self._recover_reflective_tape(
                blur, binary, inside, threshold)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if int(binary.max()) == 0:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # 普通循迹继续只使用近处 ROI；直角预览单独看更高的区域。这样
        # 远处横臂可以提前被看见，却不会把桌椅或相邻赛道送进转向拟合。
        corner = self._detect_l_corner(binary, roi_top)
        if corner.get('corner_dir'):
            # The near-ROI row-span template is useful for junction geometry,
            # but an isolated L decision here is not anchored to the path the
            # vehicle is actually following.  Autonomous L turns are accepted
            # only from the bottom-anchored connected preview below.
            corner = self._detect_l_corner(
                np.zeros((1, binary.shape[1]), np.uint8), roi_top)
        if not (corner.get('corner_dir') or corner.get('junction_left') or
                corner.get('junction_right') or
                corner.get('junction_straight')):
            preview_top = min(roi_top, int(round(wh * 0.18)))
            preview_binary = self._build_corner_preview(
                work, preview_top, threshold)
            preview_corner = self._detect_piecewise_corner(
                preview_binary, preview_top, roi_top, wh,
                self._prev_cx if self._prev_cx is not None else center)
            if preview_corner.get('corner_dir'):
                corner = preview_corner

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

        # A false foreground patch can produce three adjacent scan hits while
        # a real road stripe should span a meaningful vertical part of the ROI.
        # Reject short chains before fitting and clear the track anchor so the
        # next frame can reacquire globally.
        scan_span = max(point[1] for point in points) - min(
            point[1] for point in points)
        min_scan_span = max(24, int(round(roi_h * 0.18)))
        if scan_span < min_scan_span:
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
        # 位置连续的最长点链，再做稳健直线拟合，避免端点把轨迹拉飞。
        points = self._select_continuous_path(points)
        if len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # 道路线必须延伸到近车头区域。只在 ROI 中上部出现的细长物体
        # （例如电线）即使能提供多个扫描点，也不能向底部外推成道路。
        # 这里必须检查连续轨迹本身，不能检查直线拟合的内点：急弯并不
        # 符合单一直线，稳健拟合会恰好删掉最靠近车头的弯道末端。
        track_points = list(points)
        near_y = roi_top + int(roi_h * 0.80)
        near_points = [point for point in track_points if point[1] >= near_y]
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

        # 全局稳健直线只用于识别分叉相对主干的位置；真正的控制方向用
        # 离车辆最近的一小段轨迹拟合局部切线。这样直角弯既不会被判成
        # 无效，也不会把远处直线外推到车头位置。
        main_fit_coeffs, _ = self._robust_linear_fit(track_points)
        fit_coeffs, fit_points = self._fit_local_direction(track_points)
        if fit_coeffs is None:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)
        if main_fit_coeffs is None:
            main_fit_coeffs = fit_coeffs

        split_candidates = self._detect_split_branches(
            branch_mask, roi_top, ww, main_fit_coeffs, track_points)
        # A preview L and a near-field split are independent observations.
        # Never combine the direction from one with the position from the
        # other: doing so moved a distant L corner to the vehicle bumper and
        # triggered the maneuver far too early.
        preview_l = bool(corner.get('corner_dir'))
        branch_candidates = (self._branch_candidates(corner, ww)
                             if preview_l else
                             (split_candidates or
                              self._branch_candidates(corner, ww)))
        chosen = None
        if split_candidates and not preview_l:
            directions = {item['direction'] for item in split_candidates}
            split_y = max(item.get('split_y', roi_top)
                          for item in split_candidates)
            targets = [item['target_x'] for item in split_candidates]
            corner = dict(corner)
            corner.update({
                'corner_point': (float(np.polyval(main_fit_coeffs, split_y)),
                                 float(split_y)),
                'corner_y_ratio': float((split_y-roi_top) /
                                        max(1, roi_h-1)),
                'corner_span': float(max(targets)-min(targets)),
                'junction_left': 'left' in directions,
                'junction_straight': 'straight' in directions,
                'junction_right': 'right' in directions,
                'junction_near': bool((split_y-roi_top) /
                                      max(1, roi_h-1) >= 0.60),
                'junction_stem_x': float(np.polyval(main_fit_coeffs, split_y)),
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
                track_points = list(chosen['points'])
                fit_points = track_points

        # 用 x = k*y + b 描述当前局部道路方向。为兼容分支选择和WebUI，
        # 系数仍保存成 (0, k, b)，但控制与绘图都不再产生二次弯曲。
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
        far_points = sum(1 for point in track_points if point[1] <= far_limit)
        close_points = sum(1 for point in track_points if point[1] >= near_limit)
        # This is only a per-frame candidate. TrafficBehavior additionally
        # requires three frames and matching forward odometry before treating
        # it as the natural end of the cross branch.
        line_end_candidate = close_points >= 2 and far_points <= 1

        result = {
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
            'points': track_points,             # 连续道路轨迹（不丢弯道点）
            'fit_y_range': (None if not fit_points else
                            (float(min(point[1] for point in fit_points)),
                             float(max(point[1] for point in fit_points)))),
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
        }
        return self._check_otsu_contrast(work, result)

    @staticmethod
    def _measure_side_contrast(work, result):
        """Check full run boundaries against the unthresholded working image."""
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        binary = result['binary']
        measurements = []
        for x, y, _ in result.get('points', []):
            y, x = int(round(y)), int(round(x))
            ry = y - result['roi_top']
            if not (0 <= ry < binary.shape[0] and 0 <= x < width):
                continue
            row = binary[ry]
            if not row[x]:
                continue
            left, right = x, x + 1
            while left > 0 and row[left - 1]:
                left -= 1
            while right < width and row[right]:
                right += 1
            run_width = right - left
            band = max(3, min(8, run_width // 4))
            if left < band + 1 or right + band + 1 > width:
                continue  # An unseen side is not positive tape evidence.
            patch = gray[max(0, y - 1):min(height, y + 2)]
            middle = patch[:, left + run_width // 4:
                           max(left + run_width // 4 + 1, right - run_width // 4)]
            tape = float(np.median(middle))
            a = float(np.median(patch[:, left - band - 1:left - 1]))
            b = float(np.median(patch[:, right + 1:right + band + 1]))
            measurements.append((y, a - tape, b - tape))
        return measurements

    def _check_otsu_contrast(self, work, result):
        # Otsu can split plain shaded floor into two classes without any tape.
        # Require two brighter sides on enough rows, tolerating local glare.
        if not result['is_valid'] or self.binary_mode != 'otsu' or self.polarity != 'black':
            return result
        minimum = 12.0  # Gray levels, not a physical-width calibration.
        measured = self._measure_side_contrast(work, result)
        supported = sum(min(a, b) >= minimum for _, a, b in measured)
        required = max(3, math.ceil(len(result.get('points', [])) * .40))
        diagnostics = dict(contrast_checked_rows=len(measured),
                           contrast_support_rows=supported,
                           contrast_required_rows=required,
                           contrast_min_gray=minimum)
        if supported < required:
            self._prev_cx = None
            result = self._empty_result(binary=result['binary'], roi_top=result['roi_top'])
            result.update(diagnostics, contrast_reason='insufficient-two-sided-contrast')
        else:
            result.update(diagnostics, contrast_reason='passed')
        return result

    @staticmethod
    def _side_candidate_reconnects(binary, labels, component_index,
                                    width, roi_height):
        """Keep a side-clipped ribbon when its near end is narrow and central.

        An acute turn can enter from an image edge while the near portion is
        still the same continuous tape ribbon.  Require several lower rows to
        be narrow and return to the central corridor; broad shadows do not
        satisfy this geometry and remain rejected.
        """
        start = max(0, int(round(roi_height * 0.55)))
        end = roi_height
        # Close perspective can make a genuine 40--50 mm tape occupy roughly
        # 22% of the image.  The former 18% limit rejected those curves when
        # their far end touched a transverse line and formed one wide blob.
        max_near_width = max(50.0, width * 0.23)
        central_left = width * 0.20
        central_right = width * 0.80
        hits = []
        rows = LineDetector._row_segments(labels[start:end] == component_index)
        for rel_y, runs in enumerate(rows, start):
            if not runs:
                continue
            left, right, run_width = max(runs, key=lambda run: run[2])
            center = (left + right) * 0.5
            if run_width <= max_near_width and central_left <= center <= central_right:
                hits.append(rel_y)
        return (len(hits) >= 3 and
                max(hits) - min(hits) >= max(12, int(roi_height * 0.12)))
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

    def _add_reflection_repair(self, segments, l0, rel_y, roi_top,
                               roi_height, anchor):
        """Add one plausible outer envelope for a glare-split tape row.

        The binary image is deliberately left untouched so nearby branches
        and intersections cannot be joined globally.  A synthetic merged run
        is offered only when adjacent dark fragments enclose the predicted
        track centre and their full outside width remains physically plausible.
        """
        segments = list(segments)
        if len(segments) < 2:
            return segments
        best = None
        for pair_index, (left, right) in enumerate(
                zip(segments, segments[1:])):
            gap = int(right[0] - left[1])
            outer_width = int(right[1] - left[0])
            if gap < 2 or gap > min(24, max(4, int(outer_width * 0.55))):
                continue
            center = l0 + (left[0] + right[1]) * 0.5
            # The previous centre may already be displaced a few pixels by
            # glare, so do not require it to fall exactly inside the white
            # notch.  It must still lie inside the combined outside edges.
            if not (l0 + left[0] <= anchor <= l0 + right[1]):
                continue
            if abs(center - anchor) > self.track_half * 0.45:
                continue
            if self.line_width_model is not None:
                width_mm = self._physical_line_width_mm(
                    outer_width, roi_top + rel_y, roi_top + roi_height)
                if (width_mm is None or not
                        self.line_width_model['min_width_mm'] <= width_mm <=
                        self.line_width_model['max_width_mm']):
                    continue
            elif outer_width > max(25, self.work_width * 0.18):
                continue
            score = abs(center - anchor) + gap * 0.15
            if best is None or score < best[0]:
                best = (score, pair_index, (left[0], right[1]))
        if best is not None:
            _, pair_index, merged = best
            # Once validated as one physical ribbon, do not leave its two
            # damaged halves competing with the repaired centre candidate.
            segments = (segments[:pair_index] + [merged] +
                        segments[pair_index + 2:])
        return segments

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

    @staticmethod
    def _row_segments(binary):
        """Extract exact contiguous foreground runs for all rows in one pass."""
        edges = np.diff(np.pad((binary != 0).astype(np.int8),
                               ((0, 0), (1, 1))), axis=1)
        ys, lefts = np.nonzero(edges == 1)
        _, rights = np.nonzero(edges == -1)
        rows = [[] for _ in range(binary.shape[0])]
        for y, left, right in zip(ys.tolist(), lefts.tolist(), rights.tolist()):
            rows[y].append((left, right-1, right-left))
        return rows

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
        row_segments = self._row_segments(binary)
        q2, q1, q0 = main_fit
        for rel_y in range(first_row, last_row):
            full_y = rel_y + roi_top
            main_x = float((q2*full_y + q1)*full_y + q0)
            minimum = max(self.min_seg_width,
                          int(round(3 + 5 * rel_y / max(1, roi_h-1))))
            for left, right, run_width in row_segments[rel_y]:
                if run_width < minimum:
                    continue
                x = (left + right)*0.5
                delta = x - main_x
                if abs(delta) >= separation:
                    side_points[1 if delta > 0 else -1].append(
                        (x, float(full_y), run_width))

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
    def _robust_linear_fit(points, residual_limit=8.0):
        """Batch all two-point hypotheses, preserving consensus scoring."""
        return LineDetector._batch_corner_stem_fit(points, residual_limit)

    @staticmethod
    def _fit_local_direction(points, max_points=6):
        """Fit the near-field tangent without discarding a legitimate bend."""
        if len(points) < 3:
            return None, []
        ordered = sorted(points, key=lambda point: point[1])
        local_points = ordered[-min(max_points, len(ordered)):]
        ys = np.asarray([point[1] for point in local_points],
                        dtype=np.float64)
        xs = np.asarray([point[0] for point in local_points],
                        dtype=np.float64)
        if np.ptp(ys) < 1.0:
            return None, []
        try:
            slope, intercept = np.polyfit(ys, xs, 1)
        except (ValueError, np.linalg.LinAlgError):
            return None, []
        return (0.0, float(slope), float(intercept)), local_points

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
        if not guided:
            # Hybrid path selection: retain several row candidates and choose
            # one globally smooth ribbon instead of taking the widest run per
            # row. This rejects broad floor shadows that happen to be long.
            return self._scan_shape_continuous_path(
                binary, roi_top, ww, inside, pred)
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

            segments = self._add_reflection_repair(
                segments, l0, rel_y, roi_top, roi_h, anchor)
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

    def _scan_shape_continuous_path(self, binary, roi_top, ww, inside, pred):
        """Select a globally smooth ribbon from multi-run scan candidates.

        This is deliberately pixel-space only: no camera intrinsics or metric
        width model is needed. A path is rewarded for smooth center movement
        and stable run width, and penalized for broad/edge-touching blobs.
        """
        roi_h = binary.shape[0]
        rows = np.linspace(int(roi_h * self.scan_start_ratio),
                           roi_h - 1, self.n_scan_rows).astype(int)
        row_candidates = []
        for rel_y in rows:
            allowed = np.flatnonzero(inside[rel_y])
            if allowed.size == 0:
                row_candidates.append([])
                continue
            l0, r0 = int(allowed[0]), int(allowed[-1]) + 1
            seg = binary[rel_y, l0:r0]
            fg = np.flatnonzero(seg)
            candidates = []
            if fg.size:
                runs = np.split(fg, np.flatnonzero(np.diff(fg) > 1) + 1)
                segments = [(int(run[0]), int(run[-1]) + 1)
                            for run in runs]
                row_anchor = pred if pred is not None else ww / 2.0
                segments = self._add_reflection_repair(
                    segments, l0, rel_y, roi_top, roi_h, row_anchor)
                minimum = max(self.min_seg_width,
                              round(3 + 5 * rel_y / max(1, roi_h - 1)))
                usable_width = max(1, r0 - l0)
                for start, end in segments:
                    bw = int(end - start)
                    if bw < minimum:
                        continue
                    cx = l0 + (float(start) + float(end - 1)) / 2.0
                    edge = min(cx - l0, r0 - 1 - cx)
                    blob_penalty = max(0.0, bw / usable_width - 0.24) * 80.0
                    edge_penalty = 8.0 if edge <= 1 else 0.0
                    candidates.append({
                        'cx': cx, 'bw': float(bw), 'y': rel_y + roi_top,
                        'base': blob_penalty + edge_penalty,
                    })
            row_candidates.append(candidates)

        states = []
        for row_index, candidates in enumerate(row_candidates):
            if not candidates:
                states.append([])
                continue
            current = []
            previous = states[-1] if states else []
            for candidate in candidates:
                if not previous:
                    anchor_cost = (abs(candidate['cx'] - pred) * 0.35
                                   if pred is not None else
                                   abs(candidate['cx'] - ww / 2.0) * 0.08)
                    current.append((candidate['base'] + anchor_cost, None,
                                    candidate['cx'], candidate['bw']))
                    continue
                best = None
                prev_candidates = row_candidates[row_index - 1]
                for j, state in enumerate(previous):
                    prev_cost, _, prev_cx, prev_bw = state
                    previous_candidate = prev_candidates[j]
                    dy = max(1.0, candidate['y'] - previous_candidate['y'])
                    dx = candidate['cx'] - prev_cx
                    slope_cost = min(80.0, abs(dx / dy) * 2.2)
                    width_cost = min(60.0,
                                     abs(np.log((candidate['bw'] + 1.0) /
                                                (prev_bw + 1.0))) * 18.0)
                    cost = prev_cost + candidate['base'] + slope_cost + width_cost
                    if best is None or cost < best[0]:
                        best = (cost, j, candidate['cx'], candidate['bw'])
                current.append(best)
            states.append(current)

        nonempty = [(i, state) for i, state in enumerate(states) if state]
        if not nonempty:
            return []
        last_row, last_states = nonempty[-1]
        state_index = min(range(len(last_states)), key=lambda i: last_states[i][0])
        chosen = []
        for row_index in range(last_row, -1, -1):
            if not states[row_index]:
                continue
            state = states[row_index][state_index]
            chosen.append((state[2], row_candidates[row_index][state_index]['y'],
                           int(round(state[3]))))
            parent = state[1]
            if parent is None:
                break
            state_index = parent
        chosen.reverse()
        if len(chosen) < 3:
            return []
        span = chosen[-1][1] - chosen[0][1]
        if span < max(24, int(round(roi_h * 0.18))):
            return []
        return chosen

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
        for y, segments in enumerate(self._row_segments(binary)):
            if not segments:
                continue

            # 只使用本行最长的连续前景段。直接用 xs[0]~xs[-1] 会把墙脚、
            # 阴影等互不相连的黑块合并成一条很长的“横臂”，造成假 L 弯。
            seg_left, seg_right, seg_width = max(segments, key=lambda run: run[2])
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
        stem_fit_coeffs, stem_inliers = self._batch_corner_stem_fit(
            [(float(cx), float(y), i)
             for i, (cx, y) in enumerate(zip(stem_centers, stem_ys))],
            inlier_tol)
        best_inliers = np.zeros(len(stem_rows), dtype=bool)
        for _, _, index in stem_inliers:
            best_inliers[index] = True
        required_stem_inliers = max(6, int(math.ceil(len(stem_rows) * 0.50)))
        if (stem_fit_coeffs is None or
                int(np.count_nonzero(best_inliers)) < required_stem_inliers):
            return empty
        stem_ys_fit = stem_ys[best_inliers]
        stem_centers_fit = stem_centers[best_inliers]
        _, stem_slope, stem_intercept = stem_fit_coeffs
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

    def _build_corner_preview(self, work, preview_top, tracking_threshold):
        """Threshold a wide, forward ROI used only for corner classification."""
        gray = cv2.cvtColor(work[preview_top:, :], cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        height, width = blur.shape[:2]
        center = width / 2.0
        fractions = np.linspace(max(.94, self.crop_top_frac),
                                max(.70, self.crop_bottom_frac), height)
        half_widths = fractions * width * .5
        columns = np.arange(width, dtype=np.float64)[None, :]
        inside = ((columns >= (center-half_widths)[:, None]) &
                  (columns < (center+half_widths)[:, None]))
        if self.binary_mode == 'adaptive':
            binary = self._adaptive_binary(blur, inside)
        else:
            threshold = tracking_threshold
            if threshold is None:
                values = blur[inside]
                threshold = (self.fixed_threshold if self.binary_mode == 'fixed'
                             else self._otsu(values))
            binary = self._apply_global_threshold(blur, inside, threshold)
        return cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))

    @staticmethod
    def _morphological_skeleton(binary):
        """Return a one-pixel skeleton without requiring opencv-contrib."""
        image = (binary != 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(image)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while int(image.max()):
            eroded = cv2.erode(image, element)
            opened = cv2.dilate(eroded, element)
            skeleton = cv2.bitwise_or(
                skeleton, cv2.subtract(image, opened))
            image = eroded
        return skeleton

    @staticmethod
    def _orthogonal_line_fit(points):
        values = np.asarray(points, dtype=np.float64)
        if len(values) < 2:
            return None, float('inf')
        mean = values.mean(axis=0)
        centered = values - mean
        try:
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, float('inf')
        direction = axes[0]
        if float(np.dot(direction, values[-1]-values[0])) < 0:
            direction = -direction
        residual = centered - np.outer(centered @ direction, direction)
        rms = float(np.sqrt(np.mean(np.sum(residual*residual, axis=1))))
        return direction, rms

    @staticmethod
    def _batch_corner_stem_fit(points, residual_limit):
        """Score all two-point stem models in batches, then refit once.

        Keep the existing maximum-consensus / mean-residual selection rule.
        Two points define a line directly; no least-squares solve is needed
        for each hypothesis. Batches bound the residual matrix size.
        """
        if len(points) < 3:
            return None, []
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        first, second = np.triu_indices(len(points), 1)
        usable = ys[second] != ys[first]
        first, second = first[usable], second[usable]
        if not first.size:
            return None, []
        best_score = (-1, float('-inf'))
        best_indices = np.arange(len(points))
        for offset in range(0, first.size, 256):
            i, j = first[offset:offset+256], second[offset:offset+256]
            slope = (xs[j] - xs[i]) / (ys[j] - ys[i])
            residual = np.abs(xs[None, :] - (
                xs[i, None] + slope[:, None] * (ys[None, :] - ys[i, None])))
            inside = residual <= residual_limit
            counts = inside.sum(axis=1)
            mean = np.where(inside, residual, 0.0).sum(axis=1) / np.maximum(counts, 1)
            winner = int(np.lexsort((mean, -counts))[0])
            score = (int(counts[winner]), -float(mean[winner]))
            if score[0] >= 3 and score > best_score:
                best_score = score
                best_indices = np.flatnonzero(inside[winner])
        try:
            slope, intercept = np.polyfit(ys[best_indices], xs[best_indices], 1)
        except (ValueError, np.linalg.LinAlgError):
            return None, []
        return ((0.0, float(slope), float(intercept)),
                [points[int(i)] for i in best_indices])

    def _detect_bottom_anchored_row_corner(
            self, component, preview_top, control_top, full_height, anchor_x,
            empty):
        """Recover thick L shapes whose morphological skeleton loses an arm.

        The component has already been proven to reconnect to the near field.
        Fit its incoming stem from the bottom upward, then look for an abrupt,
        one-sided row expansion.  This preserves the bottom-anchor safety rule
        while avoiding dependence on fragile one-pixel skeleton endpoints.
        """
        height, width = component.shape[:2]
        component_rows = self._row_segments(component)

        def segments_at(y):
            return [run for run in component_rows[y]
                    if run[2] >= self.min_seg_width]

        max_y = int(np.max(np.nonzero(component)[0]))
        bottom_start = max(0, max_y - max(24, int(round(height * .25))))
        bottom_rows = []
        for y in range(bottom_start, max_y + 1):
            segments = segments_at(y)
            if not segments:
                continue
            left, right, run_width = min(
                segments,
                key=lambda item: (0.0 if item[0] <= anchor_x <= item[1]
                                  else min(abs(anchor_x-item[0]),
                                           abs(anchor_x-item[1])),
                                  -item[2]))
            bottom_rows.append(((left + right) * .5, y, run_width))
        if len(bottom_rows) < max(10, int(round(height * .10))):
            return None

        preliminary_width = float(np.median(
            [item[2] for item in bottom_rows]))
        residual_limit = max(3.0, preliminary_width * .20)
        fit, inliers = self._batch_corner_stem_fit(
            bottom_rows, residual_limit=residual_limit)
        if fit is None or len(inliers) < max(10, int(len(bottom_rows) * .60)):
            return None
        _, stem_slope, stem_intercept = fit
        if abs(stem_slope) > 1.10:
            return None
        residuals = [abs(cx - (stem_slope*y + stem_intercept))
                     for cx, y, _ in bottom_rows]
        stem_rows = [row for row, residual in zip(bottom_rows, residuals)
                     if residual <= residual_limit]
        if not stem_rows:
            return None
        normal_width = float(np.median([item[2] for item in stem_rows]))
        stem_rms = float(np.sqrt(np.mean([
            (cx-(stem_slope*y+stem_intercept))**2
            for cx, y, _ in stem_rows])))
        if stem_rms > residual_limit:
            return None

        rows = []
        upper_limit = max_y - max(12, int(round(height * .12)))
        for y in range(0, upper_limit + 1):
            predicted_x = stem_slope*y + stem_intercept
            segments = segments_at(y)
            if not segments:
                continue
            # A reflection can split the thick horizontal arm into two runs
            # on a row even though adjacent rows keep the whole tape in one
            # connected component.  Aggregate the already-anchored component
            # instead of silently selecting only the short stem-side run.
            left = min(item[0] for item in segments)
            right = max(item[1] for item in segments)
            run_width = right-left+1
            gap = min(0.0 if item[0] <= predicted_x <= item[1] else
                      min(abs(predicted_x-item[0]), abs(predicted_x-item[1]))
                      for item in segments)
            if gap <= max(5.0, normal_width * .65):
                rows.append((y, left, right, run_width, predicted_x, gap))
        if not rows:
            return None
        arm = max(rows, key=lambda item: item[3])
        arm_y, arm_left, arm_right, span, stem_x, _ = arm
        span_gate = max(width * .18, normal_width * 3.0)
        if span < span_gate:
            return None

        # A true taped corner stays broad for several adjacent rows; a single
        # noisy scanline or compression scar must not create an L candidate.
        support_radius = max(4, int(round(height * .05)))
        broad_support = [item for item in rows
                         if abs(item[0]-arm_y) <= support_radius and
                         item[3] >= span_gate * .72]
        if len(broad_support) < 3:
            return None

        # Width must expand abruptly from the incoming stem. Smooth curves
        # broaden progressively and should remain under normal line tracking.
        lower_gap = max(8, int(round(height * .05)))
        lower_band = [item[3] for item in rows
                      if arm_y + lower_gap <= item[0] <=
                      arm_y + lower_gap + max(14, int(round(height * .12)))]
        if not lower_band or span < float(np.median(lower_band)) * 2.8:
            return None

        # A vertical continuation above the arm makes this a T/cross rather
        # than an autonomous L turn.
        upper_gap = max(6, int(round(height * .04)))
        upper_stem = [item for item in rows
                      if item[0] <= arm_y-upper_gap and
                      item[5] <= normal_width*.45]
        if len(upper_stem) >= max(5, int(round(height * .06))):
            return None

        left_extent = stem_x - arm_left
        right_extent = arm_right - stem_x
        min_arm = width * .10
        margin = max(8.0, normal_width * .50)
        if left_extent >= min_arm and right_extent >= min_arm:
            return None
        if right_extent >= min_arm and right_extent >= left_extent + margin:
            direction = 1
        elif left_extent >= min_arm and left_extent >= right_extent + margin:
            direction = -1
        else:
            return None

        absolute_y = float(arm_y + preview_top)
        result = dict(empty)
        result.update({
            'corner_dir': direction,
            'corner_point': (float(stem_x), absolute_y),
            'corner_y_ratio': float(np.clip(
                (absolute_y-float(control_top)) /
                max(1.0, float(full_height-control_top)), 0.0, 1.0)),
            'corner_span': float(span),
        })
        return result

    @staticmethod
    def _reconnect_preview_corner_arm(labels, stats, stem_index):
        """Bridge one short, one-sided gap at the top of the incoming tape.

        This is confined to L classification in the wide preview.  The
        ordinary tracking mask and its metric-width guard stay untouched.
        Ambiguous or distant horizontal components are never joined.
        """
        height, width = labels.shape
        stem = labels == stem_index
        _, stem_top, _, _, _ = stats[stem_index]
        tip_band = stem[int(stem_top):min(height, int(stem_top) + 14)]
        _, tip_xs = np.nonzero(tip_band)
        if tip_xs.size < 4:
            return None
        tip_x = float(np.median(tip_xs))
        max_gap = max(8, int(round(width * .07)))
        arm_candidates = []
        for index in range(1, len(stats)):
            if index == stem_index:
                continue
            x, y, bw, bh, area = (int(value) for value in stats[index])
            if (area < 60 or bw < max(width * .18, bh * 3.0) or
                    bh > height * .15 or y + bh >= height * .78 or
                    abs(y + bh * .5 - stem_top) > height * .12):
                continue
            left_extent = tip_x - x
            right_extent = x + bw - 1 - tip_x
            if (left_extent >= width * .10 and
                    right_extent >= width * .10):
                continue                  # disconnected T/cross arm
            if max(left_extent, right_extent) < width * .18:
                continue
            arm_candidates.append(index)
        if not arm_candidates:
            return None

        distances = cv2.distanceTransform(
            (~stem).astype(np.uint8), cv2.DIST_L2, 5)
        plausible = []
        for index in arm_candidates:
            arm = labels == index
            arm_distances = np.where(arm, distances, np.inf)
            ay, ax = np.unravel_index(
                int(np.argmin(arm_distances)), labels.shape)
            if arm_distances[ay, ax] > max_gap:
                continue
            y0, y1 = max(0, ay-max_gap-2), min(height, ay+max_gap+3)
            x0, x1 = max(0, ax-max_gap-2), min(width, ax+max_gap+3)
            ys, xs = np.nonzero(stem[y0:y1, x0:x1])
            if ys.size == 0:
                continue
            ys, xs = ys+y0, xs+x0
            nearest = int(np.argmin((ys-ay)**2 + (xs-ax)**2))
            sy, sx = int(ys[nearest]), int(xs[nearest])
            if sy > stem_top + max(14, int(round(height * .12))):
                continue                  # nearby side mark, not the knee
            plausible.append((index, (ax, ay), (sx, sy)))
        if len(plausible) != 1:
            return None                  # competing arms are ambiguous

        index, arm_point, stem_point = plausible[0]
        repaired = (stem | (labels == index)).astype(np.uint8) * 255
        cv2.line(repaired, arm_point, stem_point, 255, 3)
        return repaired

    def _detect_piecewise_corner(self, binary, preview_top, control_top,
                                 full_height, anchor_x):
        """Detect a concentrated heading change on the connected tape path.

        Unlike the legacy row-span template this is rotation invariant: the
        outgoing arm may appear horizontal or diagonal under perspective.
        """
        empty = self._detect_l_corner(np.zeros((1, binary.shape[1]),
                                                np.uint8), control_top)
        height, width = binary.shape[:2]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        candidates = []
        bottom_start = int(round(height * .78))
        for index in range(1, count):
            x, y, bw, bh, area = stats[index]
            if area < 40 or bh < height * .35:
                continue
            ys, xs = np.nonzero(labels == index)
            near = ys >= bottom_start
            if not np.any(near):
                continue
            near_error = float(np.min(np.abs(xs[near]-float(anchor_x))))
            candidates.append((near_error, -int(area), index))
        if not candidates:
            return empty
        _, _, component_index = min(candidates)
        component = (labels == component_index).astype(np.uint8) * 255
        row_corner = self._detect_bottom_anchored_row_corner(
            component, preview_top, control_top, full_height, anchor_x, empty)
        if row_corner is not None:
            return row_corner
        repaired = self._reconnect_preview_corner_arm(
            labels, stats, component_index)
        if repaired is not None:
            row_corner = self._detect_bottom_anchored_row_corner(
                repaired, preview_top, control_top, full_height, anchor_x,
                empty)
            if row_corner is not None:
                return row_corner
        skeleton = self._morphological_skeleton(component)
        coords = [tuple(value) for value in np.argwhere(skeleton != 0)]
        if len(coords) < 40:
            return empty
        coord_set = set(coords)

        def neighbours(point):
            y, x = point
            return [(y+dy, x+dx) for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1) if (dy or dx) and
                    (y+dy, x+dx) in coord_set]

        max_y = max(point[0] for point in coords)
        start = min((point for point in coords if point[0] >= max_y-3),
                    key=lambda point: abs(point[1]-float(anchor_x)))
        queue = [start]
        parent = {start: None}
        distance = {start: 0}
        for point in queue:
            for nxt in neighbours(point):
                if nxt in distance:
                    continue
                distance[nxt] = distance[point] + 1
                parent[nxt] = point
                queue.append(nxt)
        endpoints = [point for point in distance
                     if len(neighbours(point)) <= 1 and point != start]
        if not endpoints:
            endpoints = [max(distance, key=distance.get)]
        farthest = max(endpoints, key=lambda point: distance[point])
        far_distance = distance[farthest]
        if far_distance < max(55.0, height * .38):
            return empty

        # Two comparably long far endpoints mean a T/cross junction. Do not
        # turn a branch choice into an autonomous L maneuver.
        long_ends = [point for point in endpoints
                     if distance[point] >= far_distance * .58]
        distinct = []
        for point in sorted(long_ends, key=lambda p: distance[p], reverse=True):
            if all(math.hypot(point[0]-other[0], point[1]-other[1]) >= 24
                   for other in distinct):
                distinct.append(point)
        if len(distinct) >= 2:
            return empty

        path = []
        point = farthest
        while point is not None:
            path.append(point)
            point = parent.get(point)
        path.reverse()
        # Convert (row, column) to (x, y), and thin graph stair-steps before
        # testing candidate breakpoints.
        xy = np.asarray([(point[1], point[0]) for point in path],
                        dtype=np.float64)
        if len(xy) < 30:
            return empty
        stride = max(1, len(xy) // 90)
        sampled = xy[::stride]
        if not np.array_equal(sampled[-1], xy[-1]):
            sampled = np.vstack((sampled, xy[-1]))
        _, single_rms = self._orthogonal_line_fit(sampled)
        minimum = max(8, len(sampled) // 6)
        best = None
        for split in range(minimum, len(sampled)-minimum):
            first = sampled[:split+1]
            second = sampled[split:]
            first_dir, first_rms = self._orthogonal_line_fit(first)
            second_dir, second_rms = self._orthogonal_line_fit(second)
            if first_dir is None or second_dir is None:
                continue
            dot = float(np.clip(np.dot(first_dir, second_dir), -1.0, 1.0))
            turn_deg = math.degrees(math.acos(dot))
            if not 55.0 <= turn_deg <= 125.0:
                continue
            piece_rms = math.sqrt(
                (first_rms**2*len(first) + second_rms**2*len(second)) /
                (len(first)+len(second)))
            score = piece_rms + .03*abs(split-len(sampled)*.5)
            if best is None or score < best[0]:
                best = (score, split, first_dir, second_dir,
                        first_rms, second_rms, piece_rms, turn_deg)
        if best is None:
            return empty
        _, split, first_dir, second_dir, first_rms, second_rms, piece_rms, _ = best
        straight_limit = max(2.8, height * .018)
        if (first_rms > straight_limit or second_rms > straight_limit or
                piece_rms >= single_rms * .58):
            return empty
        cross = (float(first_dir[0]*second_dir[1] -
                       first_dir[1]*second_dir[0]))
        if abs(cross) < .70:
            return empty
        direction = 1 if cross > 0 else -1
        knee_x, knee_y = sampled[split]
        absolute_y = float(knee_y + preview_top)
        y_ratio = ((absolute_y-float(control_top)) /
                   max(1.0, float(full_height-control_top)))
        result = dict(empty)
        result.update({
            'corner_dir': direction,
            'corner_point': (float(knee_x), absolute_y),
            'corner_y_ratio': float(np.clip(y_ratio, 0.0, 1.0)),
            'corner_span': float(np.linalg.norm(sampled[-1]-sampled[split])),
        })
        return result

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

    def _recover_reflective_tape(self, gray, strict_binary, inside,
                                 threshold, margin=30, max_distance=24):
        """Recover weak-dark tape pixels without globally joining roads.

        Reflection can lift part of black tape just above Otsu's strict
        threshold. A relaxed pixel is admitted only if its relaxed connected
        component contains strict tape and it stays near that strict seed.
        """
        if self.polarity != 'black' or int(strict_binary.max()) == 0:
            return strict_binary
        relaxed_threshold = int(min(245, int(threshold) + int(margin)))
        weak = ((gray < relaxed_threshold) & inside).astype(np.uint8)
        if int(weak.max()) == 0:
            return strict_binary

        _, labels = cv2.connectedComponents(weak, connectivity=8)
        seed_labels = np.unique(labels[strict_binary != 0])
        seed_labels = seed_labels[seed_labels != 0]
        if seed_labels.size == 0:
            return strict_binary
        connected = np.isin(labels, seed_labels)
        distance = cv2.distanceTransform(
            (strict_binary == 0).astype(np.uint8), cv2.DIST_L2, 3)
        # Only fill concavities inside the envelope of an already accepted
        # strict component. This rejects weak scratches protruding from the
        # tape even if they touch it in the relaxed mask.
        hull_support = strict_binary.copy()
        contours, _ = cv2.findContours(
            strict_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3 and cv2.contourArea(contour) > 0:
                cv2.fillConvexPoly(
                    hull_support, cv2.convexHull(contour), 255)
        # A reflection may erase the *end* of the near-field stem, which lies
        # outside its strict convex hull. Permit a short, one-way continuation
        # toward the image bottom, seeded only by strict pixels in the lower
        # part of the ROI. It cannot grow upward into an L arm or sideways into
        # another road.
        near_start = int(round(strict_binary.shape[0] * 0.55))
        near_seed = strict_binary.copy()
        near_seed[:near_start] = 0
        # Require a genuinely vertical stem. A thin horizontal arm has only a
        # few pixels per column and must not be projected toward the vehicle.
        min_vertical_support = max(
            6, int(round(strict_binary.shape[0] * 0.10)))
        stem_columns = (np.count_nonzero(near_seed, axis=0) >=
                        min_vertical_support)
        stem_seed = near_seed.copy()
        stem_seed[:, ~stem_columns] = 0
        reach = min(int(max_distance), strict_binary.shape[0] - 1)
        downward_support = cv2.dilate(
            stem_seed, np.ones((reach + 1, 1), np.uint8),
            anchor=(0, reach), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        hull_support = cv2.bitwise_or(hull_support, downward_support)
        weak_recovery = (connected & (distance <= float(max_distance)) &
                         inside & (hull_support != 0))
        # The last few centimetres may be saturated almost to floor brightness,
        # so weak thresholding has no evidence left. A vertically supported
        # stem may still be continued geometrically for this short distance.
        end_recovery = ((downward_support != 0) &
                        (distance <= float(max_distance)) & inside)
        recovered = weak_recovery | end_recovery
        binary = strict_binary.copy()
        binary[recovered] = 255
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
      corner_delay_frames 无里程反馈时的后备延迟帧数(默认10)
      corner_delay_speed  L弯延迟直行阶段的速度 mm/s(默认150)
      corner_delay_distance_m 确认L弯后按实测里程前进多少米再转(默认0.20)
      corner_turn_degrees L弯原地旋转的目标角度；越大转得越多(默认80度)
      corner_turn_speed   L弯原地旋转的目标速度 mrad/s(默认800)
      start_rotate   起步确认期间是否原地转向对准线(默认False:静止确认后边前进边修正)
    """

    def __init__(self, camera, chassis,
                 base_speed=160, max_z=800,
                 kp=12.0, kd=1.2, ka=3.5,
                 err_alpha=0.6, z_rate_limit=120.0,
                 lost_hold=10, search_frames=15,
                 startup_frames=5, ramp_frames=20,
                 corner_delay_frames=10, corner_delay_speed=150,
                 corner_delay_distance_m=0.20,
                 corner_turn_degrees=80.0, corner_turn_speed=800,
                 start_rotate=False,
                 work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, scan_start_ratio=0.25,
                 crop_bottom_frac=0.70, crop_top_frac=0.90,
                 track_half=60.0, polarity='black',
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0,
                 z_invert=True,   # 转向方向取反（默认 True）
                 target_fps=30, debug=False, web_debug=None,
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
        self.corner_delay_distance_m = max(
            0.0, float(corner_delay_distance_m))
        self.corner_turn_radians = math.radians(
            float(np.clip(corner_turn_degrees, 10.0, 180.0)))
        self.corner_turn_speed = int(np.clip(corner_turn_speed, 50, 1000))
        self.corner = CornerManeuver(
            advance_frames=self.corner_delay_frames,
            advance_speed=self.corner_delay_speed,
            advance_distance_m=self.corner_delay_distance_m,
            turn_degrees=math.degrees(self.corner_turn_radians),
            turn_speed=self.corner_turn_speed)
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
        self.corner.reset(clear_exit=True)

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
                suppress_observed = self.corner.exit_frames > 0
                corner_result = self.corner.step(
                    det, dt, enabled=self._started,
                    base_speed=self.base_speed, max_z=self.max_z,
                    z_invert=self.z_invert,
                    yaw_total_deg=self.odometry.snapshot().get(
                        'odom_yaw_total_deg'),
                    odom_distance_m=self.odometry.snapshot().get(
                        'odom_distance_m'))
                observed_corner = 0 if suppress_observed else detected_corner
                self._corner_dir = self.corner.direction
                self._corner_frames = self.corner.frames
                self._corner_phase = self.corner.phase
                self._corner_turn_radians = self.corner.turn_radians
                self._corner_exit_frames = self.corner.exit_frames

                corner_handled = (corner_result is not None and
                                  corner_result.command is not None)
                if corner_handled:
                    speed, _, z = corner_result.command
                    state = corner_result.state
                    self._lost_count = 0
                    self._has_prev = False
                    self._filtered_err = 0.0
                    self._filtered_angle = 0.0
                    self._last_sign = (self.corner.direction or
                                       self._last_sign)
                    self._last_z = (-z if self.z_invert else z)
                    if self.chassis.send_speed(speed, 0, z):
                        send_fail = 0
                    else:
                        send_fail += 1

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
                        elif (corner_result is not None and
                              corner_result.command is None):
                            speed = min(
                                speed,
                                self.corner.confirmation_speed(self.base_speed))
                            state = corner_result.state
                        elif self.corner.confirming:
                            # L 候选出现后仍按当前轨迹转向，只把前进速度压到
                            # 基础速度的 50%。连续确认完成且拐点到达触发线后，
                            # CornerManeuver 才接管并进入固定前进/原地转向阶段。
                            speed = min(
                                speed,
                                self.corner.confirmation_speed(self.base_speed))
                            state = ('corner-confirm-left'
                                     if self.corner.confirm_direction < 0
                                     else 'corner-confirm-right')
                        # Edge warning: slow before a sharp path reaches the
                        # image boundary, giving the detector time to reacquire.
                        edge_ratio = abs(float(err)) / max(1.0,
                                                           self.detector.work_width * 0.5)
                        if edge_ratio >= 0.55 or abs(float(angle)) >= 18.0:
                            speed = min(speed, int(round(self.base_speed * 0.50)))

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
                        'corner_confirm_count': self.corner.confirm_count,
                        'corner_confirm_frames': self.corner.confirm_frames,
                        'corner_exit_frames': self._corner_exit_frames,
                        'corner_delay_frames': self.corner_delay_frames,
                        'corner_delay_speed': self.corner_delay_speed,
                        'corner_delay_distance_m': self.corner_delay_distance_m,
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
            fit_range = det.get('fit_y_range')
            y_top = (fit_range[0] if fit_range is not None else
                     min(p[1] for p in det['points']))
            y_bot = (fit_range[1] if fit_range is not None else
                     max(p[1] for p in det['points']))
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
