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
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# 1. 线条检测器（追踪式）
# =====================================================================
class LineDetector:
    """从画面中提取巡线路径中心与方向（处理在降采样小图上进行）。"""

    def __init__(self, work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, min_seg_width=2,
                 polarity='black',
                 crop_bottom_frac=0.25, crop_top_frac=0.60,
                 track_half=50.0, scan_start_ratio=0.25,
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0):
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

        # 上一帧车头参考行处的线中心(工作图 x)，兼作本帧搜索窗中心
        self._prev_cx = None

    # ------------------------------------------------------------------
    def process(self, frame):
        """处理一帧，返回检测结果 dict。"""
        empty = self._empty_result()
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
        points = self._scan_lines(binary, roi_top, ww, inside, pred)
        if len(points) < 3:
            # 重捕获：不受预测限制，直接在全裁切范围内找
            points = self._scan_lines(binary, roi_top, ww, inside, None)
        if len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # 道路线必须延伸到近车头区域。只在 ROI 中上部出现的细长物体
        # （例如电线）即使能提供多个扫描点，也不能向底部外推成道路。
        near_y = roi_top + int(roi_h * 0.80)
        near_points = [point for point in points if point[1] >= near_y]
        if len(near_points) < 2:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # TODO-B4【二次拟合、横向误差与方向角】
        # 用 x = q2*y^2 + q1*y + q0 描述平滑弯道；车头参考点取
        # ROI 最底行，方向角取该点切线 dx/dy = 2*q2*y + q1。
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        q2, q1, q0 = np.polyfit(ys, xs, 2)
        ref_y = roi_top + roi_h - 1
        cx_fit = float(np.clip(q2 * ref_y ** 2 + q1 * ref_y + q0,
                               0.0, ww - 1.0))
        error_px = cx_fit - ww / 2.0
        tangent = 2.0 * q2 * ref_y + q1
        angle_deg = math.degrees(math.atan(-tangent))
        if self._prev_cx is None:
            self._prev_cx = cx_fit
        else:
            self._prev_cx = 0.6 * cx_fit + 0.4 * self._prev_cx

        return {
            'is_valid': True,
            'centroid': (cx_fit, float(ref_y)),
            'error_px': float(error_px),          # 线在右 → 正 → 右转
            'angle_deg': float(angle_deg),
            'fit_coeffs': (float(q2), float(q1), float(q0)),
            **corner,
            'points': points,                   # 参与拟合的点
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
        }

    # ------------------------------------------------------------------
    def _scan_lines(self, binary, roi_top, ww, inside, pred):
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
                if best_s < 0 or end - start > best_e - best_s:
                    best_s, best_e = start, end

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
            points.append((cx, rel_y + roi_top, bw))
        return points

    def _detect_l_corner(self, binary, roi_top):
        """检测单侧横臂的 L 弯；方向 -1=左，+1=右，0=未检测到。"""
        roi_h, ww = binary.shape[:2]
        empty = {
            'corner_dir': 0,
            'corner_point': None,
            'corner_y_ratio': 0.0,
            'corner_span': 0.0,
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
        if span < max(ww * 0.16, normal_width * 2.0):
            return empty

        # 用横臂下方的主干中心确定拐点，避免把整条横臂的中点当作道路中心。
        near_limit = arm_y + max(12, int(round(roi_h * 0.35)))
        stem_rows = [item for item in lower if item[0] <= near_limit]
        if len(stem_rows) < 4:
            stem_rows = lower[:max(4, min(10, len(lower)))]
        stem_x = float(np.median([(item[1] + item[2]) * 0.5
                                  for item in stem_rows]))
        left_extent = stem_x - arm_left
        right_extent = arm_right - stem_x
        margin = max(8.0, normal_width * 0.6)
        min_arm = ww * 0.10
        if right_extent >= min_arm and right_extent >= left_extent + margin:
            direction = 1
        elif left_extent >= min_arm and left_extent >= right_extent + margin:
            direction = -1
        else:
            return empty                 # T/十字路口，不冒充 L 弯

        return {
            'corner_dir': direction,
            'corner_point': (stem_x, float(arm_y + roi_top)),
            'corner_y_ratio': float(arm_y / max(1, roi_h - 1)),
            'corner_span': float(span),
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
                 crop_bottom_frac=0.50, crop_top_frac=0.60,
                 track_half=60.0, polarity='black',
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0,
                 z_invert=True,   # 转向方向取反（默认 True）
                 target_fps=20, debug=False, web_debug=None):
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
            adaptive_block=adaptive_block, adaptive_c=adaptive_c)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1, target_fps)
        self.debug = debug
        self.web_debug = web_debug

        self._prev_err = 0.0
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._has_prev = False      # 是否已有上一帧有效误差(首帧不微分)
        self._last_z = 0.0
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
        self.fps = 0.0
        self._fps_n = 0
        self._fps_t = time.time()

    # ------------------------------------------------------------------
    def run(self, max_frames=None):
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
            while True:
                frame = self.camera.read()
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

                observed_corner = int(det.get('corner_dir', 0))
                corner_near = float(det.get('corner_y_ratio', 0.0)) >= 0.52
                if (self._corner_dir == 0 and self._started and
                        det['is_valid'] and observed_corner and corner_near):
                    if observed_corner == self._corner_confirm_dir:
                        self._corner_confirm_count += 1
                    else:
                        self._corner_confirm_dir = observed_corner
                        self._corner_confirm_count = 1
                    if self._corner_confirm_count >= 4:
                        self._corner_dir = observed_corner
                        self._corner_frames = 0
                        self._corner_phase = 'advance'
                        self._corner_turn_radians = 0.0
                        logger.info('连续4帧确认%s L 弯，跨度=%.0fpx，先低速直行越过拐点',
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
                            raw_z = 0.0
                        else:
                            self._corner_turn_radians += abs(raw_z) * dt / 1000.0
                        self._last_z = raw_z
                        z = -raw_z if self.z_invert else raw_z
                        if self.base_speed <= 0:
                            z = 0
                        speed = 0
                        state = ('corner-left' if self._corner_dir < 0
                                 else 'corner-right')
                        if turn_complete:
                            state = 'corner-stop'
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
                        if observed_corner:
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
                    self._last_z = 0.0
                    if not self._started:
                        # 起步期间失线：不前进也不旋转搜索，原地停车等线出现
                        self._start_seen = 0
                        state, z, speed = 'start-lost', 0, 0
                        self.chassis.send_speed(0, 0, 0)
                    else:
                        state, z, speed = self._handle_lost()

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
                        'corner_delay_frames': self.corner_delay_frames,
                        'corner_delay_speed': self.corner_delay_speed,
                        'corner_turn_target_deg': math.degrees(self.corner_turn_radians),
                        'corner_turn_speed': self.corner_turn_speed,
                        'corner_turn_deg': math.degrees(self._corner_turn_radians),
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
                    st = self.chassis.read_status()
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
        1) lost-hold   短暂低速直行，等线重新回进视野；
        2) lost-search 朝最后一次见线的方向旋转搜索，超过半程后自动换向，
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
        if self.base_speed <= 0:
            self.chassis.send_speed(0, 0, 0)
            return 'lost-stop', 0, 0
        hold_limit = max(0, int(self.lost_hold))
        search_limit = max(0, int(self.search_frames))

        if self._lost_count <= hold_limit:
            speed = max(0, int(round(self.base_speed * 0.25)))
            self.chassis.send_speed(speed, 0, 0)
            return 'lost-hold', 0, speed

        search_index = self._lost_count - hold_limit
        if search_index <= search_limit:
            direction = self._last_sign
            if search_index > (search_limit + 1) // 2:
                direction = -direction
            z = direction * min(abs(float(self.max_z)), 200.0)
            if self.z_invert:
                z = -z
            self.chassis.send_speed(0, 0, int(z))
            return 'lost-search', z, 0

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
