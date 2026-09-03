#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-基于视觉的黑白线循迹（学生实验版）。

USB 摄像头读取地面画面 → 提取线中心与方向 → PD 控制 → 串口底盘驱动。
适用于"摄像头装于车顶、看得远、地面干扰多、黑线连续"的场景。

用法:
    python run.py                     # 正常巡线（自动找摄像头/串口）
    python run.py --debug             # 显示调试窗口(原图+二值图+状态)
    python run.py --camera 1          # 指定摄像头编号
    python run.py --port COM3         # 指定底盘串口
    python run.py --black / --white   # 黑白极性（默认 black）
    python run.py --binary-mode fixed --threshold 100
    python run.py --binary-mode otsu  # 学生手写 Otsu（默认）
    python run.py --speed 160         # 直道巡航速度 mm/s
    python run.py --kp 2.0 --kd 1.0   # PD 增益
    python run.py --crop-bottom 0.25  # 近车头裁剪宽度比例
    python run.py --crop-top 0.60     # 远处裁剪宽度比例（转弯余量）
    python run.py --track-half 50     # 滑动搜索窗半宽 px
    python run.py --max-frames 600    # 跑 600 帧后自动停止
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import serial  # 用于捕获串口连接异常

# 保证本案例目录在 sys.path 中，可独立运行（不依赖工程公共包）
_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CASE_DIR not in sys.path:
    sys.path.insert(0, _CASE_DIR)

from comm.chassis import ChassisController
from comm.usb_camera import USBCamera
from core.line_follower import LineFollower
from debug_web import CONFIG_SCHEMA, DebugWebServer


_WEB_CONFIG_PATH = os.path.join(_CASE_DIR, 'web_config.json')


def _load_web_config(logger):
    try:
        with open(_WEB_CONFIG_PATH, 'r', encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning('无法读取网页参数 %s: %s', _WEB_CONFIG_PATH, exc)
        return {}


def _set_manual_exposure(camera, exposure, logger):
    """用 v4l2 锁定 UVC 摄像头曝光，避免遮挡时自动亮度跳变。"""
    if not isinstance(camera.device, int):
        logger.warning('当前视频源不是 V4L2 设备，跳过手动曝光')
        return False
    device_path = f'/dev/video{camera.device}'
    command = [
        'v4l2-ctl', '-d', device_path,
        '-c', 'exposure_auto=1',
        '-c', f'exposure_absolute={int(exposure)}',
        '-c', 'exposure_auto_priority=0',
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=3)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning('设置摄像头手动曝光失败: %s', exc)
        return False
    logger.info('摄像头手动曝光已锁定: device=%s exposure=%d',
                device_path, exposure)
    return True


def main():
    parser = argparse.ArgumentParser(description='04-基于视觉的黑白线循迹')
    parser.add_argument('--camera', type=int, default=None,
                        help='USB 摄像头编号（默认自动扫描 0~3）')
    parser.add_argument('--port', type=str, default=None,
                        help='底盘串口，如 COM3（默认自动识别）')
    parser.add_argument('--baud', type=int, default=115200, help='串口波特率')
    parser.add_argument('--debug', action='store_true', help='显示调试窗口')
    parser.add_argument('--web', action='store_true',
                        help='启动 localhost 实时调试网页')
    parser.add_argument('--web-host', default='127.0.0.1',
                        help='调试网页监听地址（默认127.0.0.1）')
    parser.add_argument('--web-port', type=int, default=9090,
                        help='调试网页端口（默认9090）')
    parser.add_argument('--web-fps', type=float, default=8.0,
                        help='网页图像刷新率上限（默认8 FPS）')
    parser.add_argument('--exposure', type=int, default=150,
                        help='摄像头手动曝光值（默认150）')
    parser.add_argument('--speed', type=int, default=160,
                        help='直道巡航速度 mm/s（底盘限幅 ±300）')
    parser.add_argument('--max-z', type=int, default=800,
                        help='最大转向速度 mrad/s，默认 800（约45.8度/s）')
    parser.add_argument('--black', action='store_true', help='黑线白底（默认）')
    parser.add_argument('--white', action='store_true', help='白线黑底')
    parser.add_argument('--binary-mode', choices=('fixed', 'otsu', 'adaptive'),
                        default='otsu',
                        help='二值化算法：固定阈值/手写Otsu/自适应阈值')
    parser.add_argument('--threshold', type=int, default=100,
                        help='fixed 模式的灰度阈值(0~255)')
    parser.add_argument('--adaptive-block', type=int, default=31,
                        help='adaptive 模式的局部窗口边长(奇数且>=3)')
    parser.add_argument('--adaptive-c', type=float, default=8.0,
                        help='adaptive 模式的局部均值修正常数')
    parser.add_argument('--no-z-invert', action='store_true',
                        help='转向方向不取反（默认取反）')
    parser.add_argument('--kp', type=float, default=12.0, help='横向误差比例增益(配合max_z=800)')
    parser.add_argument('--kd', type=float, default=1.2, help='横向误差微分增益')
    parser.add_argument('--ka', type=float, default=3.5, help='方向角前馈增益')
    parser.add_argument('--err-alpha', type=float, default=0.6,
                        help='误差/角度低通滤波系数 0~1（默认0.6）')
    parser.add_argument('--z-rate', type=float, default=120.0,
                        help='每帧最大转向增量 mrad/s（默认120，防猛甩）')
    parser.add_argument('--lost-hold', type=int, default=10,
                        help='失线低速直行的帧数上限')
    parser.add_argument('--search-frames', type=int, default=15,
                        help='失线后低速旋转搜索的帧数上限')
    parser.add_argument('--startup-frames', type=int, default=5,
                        help='起步确认帧数：连续检测到线这么多帧后才开始前进(默认5)')
    parser.add_argument('--ramp-frames', type=int, default=20,
                        help='起步后速度从0平滑加速到目标的帧数(默认20，约1秒)')
    parser.add_argument('--corner-delay-frames', type=int, default=10,
                        help='确认L弯后继续低速直行的帧数（越大越晚转，默认10）')
    parser.add_argument('--corner-delay-speed', type=int, default=40,
                        help='确认L弯后延迟直行阶段的速度 mm/s（默认40）')
    parser.add_argument('--corner-turn-degrees', type=float, default=78.0,
                        help='L弯原地旋转的目标角度（度，默认78）')
    parser.add_argument('--corner-turn-speed', type=int, default=300,
                        help='L弯原地旋转的目标速度 mrad/s（默认300）')
    parser.add_argument('--start-rotate', action='store_true',
                        help='起步时原地转向对准线后再前进(默认关:静止确认后边前进边修正)')
    parser.add_argument('--roi-top', type=float, default=0.45,
                        help='垂直方向检测区起点比例(0~1，越小看得越远)')
    parser.add_argument('--n-scan-rows', type=int, default=12,
                        help='扫描行数（默认12）')
    parser.add_argument('--scan-start', type=float, default=0.25,
                        help='扫描起点在ROI内的比例(0~1，默认0.25)')
    parser.add_argument('--crop-bottom', type=float, default=0.50,
                        help='近车头裁剪宽度比例(0~1，需足以容纳转向时的线，默认0.50)')
    parser.add_argument('--crop-top', type=float, default=0.60,
                        help='远处裁剪宽度比例(0~1，越大越宽=转弯余量，默认0.60)')
    parser.add_argument('--track-half', type=float, default=60.0,
                        help='滑动搜索窗半宽 px（默认60，越大越不易丢线但抗干扰弱）')
    parser.add_argument('--work-width', type=int, default=320,
                        help='处理分辨率宽度(默认320)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大运行帧数(调试用)')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s %(name)s] %(message)s')
    logger = logging.getLogger('line_follow')

    # 网页保存的参数作为默认值，显式命令行参数仍可覆盖它们。
    saved_config = _load_web_config(logger)
    parser.set_defaults(**{
        name: saved_config[name]
        for name in CONFIG_SCHEMA
        if name in saved_config
    })
    if saved_config.get('binary_mode') in ('fixed', 'otsu', 'adaptive'):
        parser.set_defaults(binary_mode=saved_config['binary_mode'])
    args = parser.parse_args()

    if args.white:
        polarity = 'white'
    elif args.black:
        polarity = 'black'
    else:
        polarity = saved_config.get('polarity', 'black')

    # 1. USB 摄像头
    camera = USBCamera(device=args.camera, width=640, height=480, fps=30)
    if not camera.open():
        logger.error('无法打开 USB 摄像头(已尝试 0~3)，请检查连接')
        return
    logger.info('摄像头已打开: device=%s size=%s', camera.device, camera.actual_size)
    _set_manual_exposure(camera, args.exposure, logger)

    # 2. 底盘串口
    ports = ChassisController.list_ports()
    logger.info('检测到串口: %s', ports if ports else '无 (请检查 USB 转串口)')
    chassis = ChassisController(port=args.port, baudrate=args.baud)
    try:
        chassis.connect()
        ch_port = chassis.port
        logger.info('底盘串口已连接: %s @ %d', ch_port, args.baud)
    except PermissionError as e:
        logger.error('串口被占用: %s（请先关闭 qt_car_monitor 等占用串口的程序）', e)
        camera.release()
        return
    except (OSError, serial.SerialException) as e:
        logger.error('底盘连接失败: %s（可尝试 --port 指定串口）', e)
        camera.release()
        return
    except ConnectionError as e:
        logger.error('底盘连接失败: %s', e)
        camera.release()
        return

    # 3. 可选的 localhost 网页调试服务
    web_debug = None
    if args.web:
        current_config = {
            name: getattr(args, name)
            for name in CONFIG_SCHEMA
        }
        current_config['binary_mode'] = args.binary_mode
        current_config['polarity'] = polarity
        web_debug = DebugWebServer(args.web_host, args.web_port,
                                   stream_fps=args.web_fps,
                                   config=current_config,
                                   config_path=_WEB_CONFIG_PATH)
        try:
            web_debug.start()
        except OSError as e:
            logger.error('调试网页启动失败: %s', e)
            chassis.close()
            camera.release()
            return

    # 4. 巡线控制器
    follower = LineFollower(
        camera, chassis,
        base_speed=args.speed,
        max_z=args.max_z,
        kp=args.kp, kd=args.kd, ka=args.ka,
        err_alpha=args.err_alpha,
        z_rate_limit=args.z_rate,
        lost_hold=args.lost_hold,
        search_frames=args.search_frames,
        startup_frames=args.startup_frames,
        ramp_frames=args.ramp_frames,
        corner_delay_frames=args.corner_delay_frames,
        corner_delay_speed=args.corner_delay_speed,
        corner_turn_degrees=args.corner_turn_degrees,
        corner_turn_speed=args.corner_turn_speed,
        start_rotate=args.start_rotate,
        work_width=args.work_width,
        roi_top_ratio=args.roi_top,
        n_scan_rows=args.n_scan_rows,
        scan_start_ratio=args.scan_start,
        crop_bottom_frac=args.crop_bottom,
        crop_top_frac=args.crop_top,
        track_half=args.track_half,
        polarity=polarity,
        binary_mode=args.binary_mode,
        fixed_threshold=args.threshold,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        z_invert=not args.no_z_invert,   # 转向方向取反（默认 True）
        debug=args.debug,
        web_debug=web_debug,
    )
    logger.info('极性=%s 二值化=%s 裁切(底%.2f/顶%.2f) 搜索窗=%gpx 转向取反=%s',
                polarity, args.binary_mode, args.crop_bottom, args.crop_top, args.track_half,
                not args.no_z_invert)

    restart_requested = False
    try:
        follower.run(max_frames=args.max_frames)
    finally:
        restart_requested = bool(web_debug and web_debug.restart_requested)
        if web_debug is not None:
            web_debug.stop()
        chassis.close()
        camera.release()
        logger.info('04 巡线程序退出，串口 %s 已释放', ch_port)

    if restart_requested:
        logger.info('正在使用网页保存的参数重启')
        os.execv(sys.executable, [
            sys.executable, os.path.abspath(__file__),
            '--web', '--web-host', args.web_host,
            '--web-port', str(args.web_port),
            '--web-fps', str(args.web_fps),
        ])


if __name__ == '__main__':
    main()
