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

底盘输出约定：x 为前进速度(mm/s)，z 为转向角速度(度/s)。
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
                 n_scan_rows=12, min_seg_width=12, max_seg_width=60,
                 polarity='black',
                 crop_bottom_frac=0.25, crop_top_frac=0.60,
                 track_half=50.0, scan_start_ratio=0.25,
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0):
        self.work_width = work_width
        self.roi_top_ratio = roi_top_ratio          # 垂直方向：只处理底部这段(车前方地面)
        self.n_scan_rows = n_scan_rows              # 扫描行数
        self.min_seg_width = min_seg_width          # 过滤过窄噪点段
        self.max_seg_width = max_seg_width          # 过滤阴影/大黑块等过宽区域
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
        if self._prev_cx is not None:
            # 弯道中线路可能很快跑出固定梯形。沿上一帧中心附加
            # 一条动态竖向走廊；预测窗仍负责抗干扰，而全窗重捕可在
            # 延迟转向后把快速横移的线找回来。
            tracking_margin = max(self.track_half * 1.5, ww * 0.12)
            tracking_corridor = (
                (columns >= self._prev_cx - tracking_margin) &
                (columns <= self._prev_cx + tracking_margin))
            inside |= tracking_corridor

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

        # 前景占比只做退化画面保护。室内阴影可能占据较大面积，不能在
        # 连通域几何过滤前直接判整帧无效，否则阴影旁的真实黑线也会丢失。
        fg_ratio = binary[inside].mean() / 255.0
        if fg_ratio < 0.005 or fg_ratio > 0.90:
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
        min_h = roi_h * 0.18          # 急弯可见纵向高度较短，仍保留横块过滤
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
            # area / height 是该组件每一行的平均宽度。大片阴影即使边缘
            # 不规则、fill 较低，平均行宽仍明显大于目标黑线。
            mean_row_width = area / float(bh_)
            if mean_row_width > self.max_seg_width:
                continue
            line_mask[labels == i] = 255
        binary = line_mask
        if int(binary.max()) == 0:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

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

        # TODO-B4【拟合、横向误差与方向角】
        # 已给出拟合输入。请完成：
        #   1) 用 np.polyfit(ys, xs, 1) 得到 x = a*y + b；
        #   2) 在 ROI 最底行 ref_y 计算 cx_fit；
        #   3) error_px = cx_fit - ww/2，线在右侧时应为正；
        #   4) 因图像 y 轴向下，方向角可由 atan(-a) 得到并转为角度；
        #   5) 用一阶 EMA 更新 self._prev_cx，供下一帧滑窗预测。
        # 验收：人工平移直线时误差符号正确，倾斜线角度符号正确且连续。
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        a, b = np.polyfit(ys, xs, 1)
        ref_y = roi_top + roi_h - 1
        cx_fit = float(np.clip(a * ref_y + b, 0.0, ww - 1.0))
        error_px = cx_fit - ww / 2.0
        angle_deg = math.degrees(math.atan(-a))
        if self._prev_cx is None:
            self._prev_cx = cx_fit
        else:
            self._prev_cx = 0.6 * cx_fit + 0.4 * self._prev_cx

        return {
            'is_valid': True,
            'centroid': (cx_fit, float(ref_y)),
            'error_px': float(error_px),          # 线在右 → 正 → 右转
            'angle_deg': float(angle_deg),
            'a': a, 'b': b,                     # 主直线参数 x=a*y+b（调试画线用）
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
                width = end - start
                if (self.min_seg_width <= width <= self.max_seg_width and
                        (best_s < 0 or width > best_e - best_s)):
                    best_s, best_e = start, end

            if best_s < 0:
                continue
            bw = best_e - best_s
            if not self.min_seg_width <= bw <= self.max_seg_width:
                continue
            cx = l0 + (best_s + best_e) // 2
            points.append((cx, rel_y + roi_top, bw))
        return points

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
            'a': 0.0, 'b': 0.0,
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
      max_z       最大转向速度 度/s
      kp / kd / ka  横向误差比例/微分/角度前馈增益
      err_alpha   误差/角度低通滤波系数(0~1)
      lost_hold   失线低速直行的帧数上限，超过则停车
      startup_frames 起步确认帧数：连续检测到线这么多帧后车辆才开始前进(默认5)
      ramp_frames    起步后速度从0平滑加速到目标的帧数(默认20，约1秒)
      start_rotate   起步确认期间是否原地转向对准线(默认False:静止确认后边前进边修正)
    """

    def __init__(self, camera, chassis,
                 base_speed=160, max_z=800,
                 kp=12.0, kd=1.2, ka=3.5,
                 err_alpha=0.6, z_rate_limit=120.0, turn_delay_frames=4,
                 lost_hold=10, search_frames=15,
                 startup_frames=5, ramp_frames=20, start_rotate=False,
                 work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, scan_start_ratio=0.25,
                 crop_bottom_frac=0.50, crop_top_frac=0.60,
                 track_half=60.0, line_min_width=12, line_max_width=60,
                 polarity='black',
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
        self.z_rate_limit = z_rate_limit    # 每帧最大转向增量(°/s)，防猛甩
        self.turn_delay_frames = max(0, int(turn_delay_frames))
        # 只对明显转弯的“新方向”做延迟，小幅回正不延迟。
        self.turn_delay_threshold = max(60.0, min(160.0, abs(max_z) * 0.18))
        self.lost_hold = lost_hold
        self.search_frames = search_frames  # 失线后低速旋转搜索的帧数
        self.z_invert = z_invert            # 转向方向取反(硬件/装向与协议约定相反时使用)
        self.startup_frames = startup_frames  # 起步确认帧数：线连续稳定这么多帧后才前进
        self.ramp_frames = ramp_frames        # 起步后速度从0平滑加速到目标所用帧数
        self.start_rotate = start_rotate      # 起步是否原地转向对准线(默认关，静止确认后前进)

        self.detector = LineDetector(
            work_width=work_width, roi_top_ratio=roi_top_ratio,
            n_scan_rows=n_scan_rows, scan_start_ratio=scan_start_ratio,
            crop_bottom_frac=crop_bottom_frac, crop_top_frac=crop_top_frac,
            track_half=track_half, min_seg_width=line_min_width,
            max_seg_width=line_max_width, polarity=polarity,
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
        self._active_turn_sign = 0
        self._pending_turn_sign = 0
        self._pending_turn_count = 0
        self.fps = 0.0
        self._fps_n = 0
        self._fps_t = time.time()

    # ------------------------------------------------------------------
    def run(self, max_frames=None):
        """主循环。"""
        logger.info('巡线启动: 极性=%s base=%dmm/s max_z=%d°/s',
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
                turn_waiting = False

                if det['is_valid']:
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
                    z_target = float(np.clip(
                        z_raw, -abs(self.max_z), abs(self.max_z)))

                    # 转向触发延迟：大转向或 S 弯换向时，新方向必须
                    # 稳定持续 turn_delay_frames 帧才生效。当需求回到小幅时
                    # 立即退出弯道状态，不把出弯回正也一起延迟。
                    demand_sign = (1 if z_target > 0 else -1
                                   if z_target < 0 else 0)
                    is_clear_turn = abs(z_target) >= self.turn_delay_threshold
                    if self.turn_delay_frames <= 0 or not is_clear_turn:
                        z = z_target
                        if not is_clear_turn:
                            self._active_turn_sign = 0
                            self._pending_turn_sign = 0
                            self._pending_turn_count = 0
                    elif demand_sign == self._active_turn_sign:
                        z = z_target
                        self._pending_turn_sign = 0
                        self._pending_turn_count = 0
                    else:
                        if demand_sign == self._pending_turn_sign:
                            self._pending_turn_count += 1
                        else:
                            self._pending_turn_sign = demand_sign
                            self._pending_turn_count = 1
                            # S 弯换向时先立即撤掉旧方向，不继续打反向。
                            self._last_z = 0.0
                        if self._pending_turn_count >= self.turn_delay_frames:
                            self._active_turn_sign = demand_sign
                            self._pending_turn_sign = 0
                            self._pending_turn_count = 0
                            z = z_target
                        else:
                            z = 0.0
                            turn_waiting = True

                    # 转向速率限制：单帧最多变化 z_rate_limit °/s，防车身猛甩
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
                        # 用“当前检测到的转向需求”降速，而不是延迟后的
                        # z。因此即使还在 turn-delay，车也会先慢下来。
                        turn_ratio = min(1.0, abs(z_target) / max_turn)
                        curve_scale = max(0.3, 1.0 - 0.7 * turn_ratio)
                        if turn_waiting:
                            curve_scale = min(curve_scale, 0.5)
                        speed = int(round(self.base_speed * ramp * curve_scale))
                        if abs(err) > 40:
                            speed = min(speed, int(round(self.base_speed * 0.3)))
                        if self.chassis.send_speed(speed, 0, int(z)):
                            send_fail = 0
                        else:
                            send_fail += 1
                        self._prev_err = err
                        if turn_waiting:
                            state = 'turn-delay'
                else:
                    self._has_prev = False
                    self._filtered_err = 0.0
                    self._filtered_angle = 0.0
                    self._last_z = 0.0
                    self._active_turn_sign = 0
                    self._pending_turn_sign = 0
                    self._pending_turn_count = 0
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
                    point_widths = [point[2] for point in det.get('points', [])]
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
                        'turn_delay_frames': self.turn_delay_frames,
                        'turn_delay_count': self._pending_turn_count,
                        'line_width_min': min(point_widths) if point_widths else None,
                        'line_width_avg': (sum(point_widths) / len(point_widths)) if point_widths else None,
                        'line_width_max': max(point_widths) if point_widths else None,
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
                        real_dps = f" 实转z={st['real_z'] * 57.3:+.0f}°/s"  # mrad/s -> °/s
                        real_dps += f" 实x/y={st['real_x']:+d}/{st['real_y']:+d}mm/s"
                        real_dps += f" 陀螺z={st['ang_vel_z']:+d}"
                    logger.info(
                        '状态=%-8s 极性=%-5s | err=%6.1fpx 角度=%5.1f° | '
                        '速度=%3dmm/s 转向=%4d°/s | FPS=%.0f | 失帧=%d%s',
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
            x_top = int((det['a'] * y_top + det['b']) * s)
            x_bot = int((det['a'] * y_bot + det['b']) * s)
            cv2.line(disp, (x_top, int(y_top * s)),
                     (x_bot, int(y_bot * s)), (0, 255, 0), 2)
            cx, cy = det['centroid']
            cv2.circle(disp, (int(cx * s), int(cy * s)), 6, (0, 255, 255), -1)

        info = (f"state={state} x={speed}mm/s z={z:.0f}d/s | "
                f"err={det['error_px']:.1f}px ang={det['angle_deg']:.1f}deg | "
                f"polar={det['line_type']} fps={self.fps:.0f}")
        cv2.putText(disp, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.imshow('LineFollower (04)', disp)
