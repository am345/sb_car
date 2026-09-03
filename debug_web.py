#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local, dependency-free web dashboard for the line follower."""

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# 网页允许修改的参数与安全范围。类型、范围在服务端再校验，
# 不信任浏览器提交的内容。
CONFIG_SCHEMA = {
    'speed': (int, 0, 300),
    'max_z': (int, 0, 1500),
    'kp': (float, 0.0, 50.0),
    'kd': (float, 0.0, 20.0),
    'ka': (float, 0.0, 20.0),
    'err_alpha': (float, 0.01, 1.0),
    'z_rate': (float, 1.0, 500.0),
    'roi_top': (float, 0.0, 0.9),
    'scan_start': (float, 0.0, 0.9),
    'crop_bottom': (float, 0.1, 1.0),
    'crop_top': (float, 0.1, 1.0),
    'track_half': (float, 5.0, 160.0),
    'startup_frames': (int, 1, 100),
    'ramp_frames': (int, 0, 200),
    'lost_hold': (int, 0, 100),
    'search_frames': (int, 0, 200),
    'threshold': (int, 0, 255),
    'adaptive_block': (int, 3, 151),
    'adaptive_c': (float, -50.0, 50.0),
}
CONFIG_ENUMS = {
    'binary_mode': {'fixed', 'otsu', 'adaptive'},
    'polarity': {'black', 'white'},
}


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>视觉循迹实时调试</title>
  <style>
    :root { color-scheme: dark; --bg:#0a0f18; --card:#111a28; --line:#25344a;
      --text:#e8eef8; --muted:#8fa2bb; --cyan:#35d4e8; --green:#50dc8b;
      --yellow:#ffc857; --red:#ff6577; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at top,#142238,var(--bg) 46%);
      color:var(--text); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { width:min(1500px,96vw); margin:20px auto; }
    header { display:flex; align-items:end; justify-content:space-between; gap:16px; margin-bottom:14px; }
    h1 { margin:0; font-size:clamp(20px,3vw,32px); letter-spacing:.04em; }
    .sub { color:var(--muted); margin-top:4px; }
    #connection { border:1px solid var(--line); border-radius:999px; padding:6px 12px; }
    .layout { display:grid; grid-template-columns:minmax(0,2fr) minmax(290px,1fr); gap:14px; }
    .card { background:color-mix(in srgb,var(--card) 94%,transparent); border:1px solid var(--line);
      border-radius:14px; box-shadow:0 16px 45px #0006; overflow:hidden; }
    .video-head { padding:10px 14px; display:flex; justify-content:space-between; color:var(--muted); }
    #stream { display:block; width:100%; min-height:260px; background:#05080d; object-fit:contain; }
    .metrics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; padding:12px; }
    .metric { padding:12px; border:1px solid var(--line); border-radius:10px; background:#0b1320; }
    .metric.wide { grid-column:1/-1; }
    .label { color:var(--muted); font-size:12px; }
    .value { margin-top:2px; font:600 24px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .value.small { font-size:17px; }
    .ok { color:var(--green); } .warn { color:var(--yellow); } .bad { color:var(--red); }
    .bars { display:grid; gap:10px; padding:0 12px 12px; }
    .bar-label { display:flex; justify-content:space-between; color:var(--muted); margin-bottom:4px; }
    .track { height:9px; overflow:hidden; background:#070c13; border-radius:99px; border:1px solid var(--line); }
    .fill { height:100%; width:50%; background:linear-gradient(90deg,var(--cyan),var(--green)); transition:width .15s; }
    .config { margin-top:14px; padding:16px; }
    .config h2 { margin:0 0 4px; font-size:20px; }
    .config-note { color:var(--muted); margin-bottom:14px; }
    .form-grid { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:12px; }
    .field label { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    input,select { width:100%; border:1px solid var(--line); border-radius:8px; padding:9px 10px;
      color:var(--text); background:#08111e; font:15px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .actions { display:flex; align-items:center; gap:12px; margin-top:16px; }
    button { border:0; border-radius:9px; padding:10px 18px; background:var(--cyan); color:#041016;
      font-weight:700; cursor:pointer; }
    button:disabled { opacity:.5; cursor:wait; }
    #saveResult { color:var(--muted); }
    footer { color:var(--muted); margin-top:10px; text-align:right; }
    @media (max-width:850px) { .layout { grid-template-columns:1fr; } .form-grid { grid-template-columns:repeat(2,1fr); } }
    @media (max-width:520px) { .form-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <header><div><h1>视觉循迹实时调试</h1><div class="sub">RAW + 拟合结果 / BINARY 检测区</div></div>
    <div id="connection" class="warn">正在连接…</div></header>
  <section class="layout">
    <div class="card"><div class="video-head"><span>实时画面</span><span id="frame">frame --</span></div>
      <img id="stream" src="/stream.mjpg" alt="debug stream"></div>
    <aside class="card">
      <div class="metrics">
        <div class="metric wide"><div class="label">运行状态</div><div id="state" class="value small">--</div></div>
        <div class="metric"><div class="label">线检测</div><div id="valid" class="value">--</div></div>
        <div class="metric"><div class="label">FPS</div><div id="fps" class="value">--</div></div>
        <div class="metric"><div class="label">横向误差</div><div id="error" class="value">--</div></div>
        <div class="metric"><div class="label">方向角</div><div id="angle" class="value">--</div></div>
        <div class="metric"><div class="label">前进速度</div><div id="speed" class="value">--</div></div>
        <div class="metric"><div class="label">转向速度</div><div id="turn" class="value">--</div></div>
      </div>
      <div class="bars">
        <div><div class="bar-label"><span>转向负载</span><span id="turnPct">--</span></div><div class="track"><div id="turnBar" class="fill"></div></div></div>
        <div><div class="bar-label"><span>起步确认</span><span id="startCount">--</span></div><div class="track"><div id="startBar" class="fill"></div></div></div>
        <div class="metric"><div class="label">检测点 / 失线 / 相机无帧</div><div id="counts" class="value small">--</div></div>
        <div class="metric"><div class="label">P / D / 角度前馈</div><div id="terms" class="value small">--</div></div>
        <div class="metric"><div class="label">二值化 / 线极性</div><div id="mode" class="value small">--</div></div>
      </div>
    </aside>
  </section>
  <section class="card config">
    <h2>运行参数</h2>
    <div class="config-note">保存后程序会先停车并释放设备，再用新参数自动重启。调速前请确保车轮架空或周围无人。</div>
    <form id="configForm"><div id="formGrid" class="form-grid"></div>
      <div class="actions"><button id="saveButton" type="submit">保存并重启</button><span id="saveResult"></span></div>
    </form>
  </section>
  <footer>数据每 200 ms 刷新；图像为 MJPEG 实时流。</footer>
</main>
<script>
const $=id=>document.getElementById(id), num=(v,n=1)=>Number(v||0).toFixed(n);
const fields=[
 ['speed','巡航速度 mm/s','number','1'],['max_z','最大转向 °/s','number','1'],
 ['kp','P 增益','number','0.1'],['kd','D 增益','number','0.1'],['ka','角度前馈','number','0.1'],
 ['err_alpha','滤波系数','number','0.05'],['z_rate','转向变化限制','number','1'],
 ['roi_top','ROI 起点','number','0.05'],['scan_start','扫描起点','number','0.05'],
 ['crop_bottom','近处宽度比','number','0.05'],['crop_top','远处宽度比','number','0.05'],
 ['track_half','搜索窗半宽 px','number','1'],['startup_frames','起步确认帧','number','1'],
 ['ramp_frames','加速斜坡帧','number','1'],['lost_hold','失线保持帧','number','1'],
 ['search_frames','失线搜索帧','number','1'],['threshold','固定阈值','number','1'],
 ['adaptive_block','自适应邻域','number','2'],['adaptive_c','自适应 C','number','0.5'],
 ['binary_mode','二值化','select',['otsu','fixed','adaptive']],
 ['polarity','线路极性','select',['black','white']]
];
function buildForm(c){
 const grid=$('formGrid'); grid.innerHTML='';
 for(const [name,label,type,extra] of fields){
  const box=document.createElement('div'); box.className='field';
  const lab=document.createElement('label'); lab.textContent=label; lab.htmlFor='cfg_'+name; box.appendChild(lab);
  let input;
  if(type==='select'){ input=document.createElement('select'); for(const v of extra){const o=document.createElement('option');o.value=v;o.textContent=v;input.appendChild(o);} }
  else { input=document.createElement('input'); input.type='number'; input.step=extra; }
  input.id='cfg_'+name; input.name=name; input.value=c[name]??''; box.appendChild(input); grid.appendChild(box);
 }
}
async function loadConfig(){ const r=await fetch('/api/config',{cache:'no-store'}); buildForm(await r.json()); }
async function refresh(){
  try {
    const r=await fetch('/api/status',{cache:'no-store'}); if(!r.ok) throw Error(r.status);
    const d=await r.json(), age=Date.now()/1000-d.updated_at;
    $('connection').textContent=age<2?'已连接':`数据延迟 ${num(age)}s`;
    $('connection').className=age<2?'ok':'warn';
    $('state').textContent=d.state||'等待数据';
    $('valid').textContent=d.valid?'有效':'无效'; $('valid').className='value '+(d.valid?'ok':'bad');
    $('fps').textContent=num(d.fps); $('error').textContent=`${num(d.error_px)} px`;
    $('angle').textContent=`${num(d.angle_deg)}°`; $('speed').textContent=`${num(d.speed,0)} mm/s`;
    $('turn').textContent=`${num(d.turn,0)}°/s`; $('frame').textContent=`frame ${d.frame_count??'--'}`;
    const tp=Math.min(100,Math.abs(d.turn||0)/Math.max(1,d.max_z||1)*100);
    $('turnPct').textContent=`${num(tp,0)}%`; $('turnBar').style.width=tp+'%';
    const sp=d.started?100:Math.min(100,(d.start_seen||0)/Math.max(1,d.startup_frames||1)*100);
    $('startBar').style.width=sp+'%'; $('startCount').textContent=d.started?'已起步':`${d.start_seen||0}/${d.startup_frames||0}`;
    $('counts').textContent=`${d.point_count||0} / ${d.lost_count||0} / ${d.no_frame_count||0}`;
    $('terms').textContent=`${num(d.p_term)} / ${num(d.d_term)} / ${num(d.angle_term)}`;
    $('mode').textContent=`${d.binary_mode||'--'} / ${d.polarity||'--'}`;
  } catch(e) { $('connection').textContent='连接中断'; $('connection').className='bad'; }
}
refresh(); setInterval(refresh,200);
$('configForm').addEventListener('submit',async e=>{
 e.preventDefault(); const button=$('saveButton'), result=$('saveResult'); button.disabled=true; result.textContent='正在保存…';
 const data={}; for(const [name,,type] of fields){const v=$('cfg_'+name).value; data[name]=type==='number'?Number(v):v;}
 try { const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status); result.textContent='已保存，程序正在停车并重启…';
  setTimeout(()=>location.reload(),2500);
 } catch(err){ result.textContent='保存失败：'+err.message; button.disabled=false; }
});
loadConfig().catch(e=>$('saveResult').textContent='读取参数失败：'+e.message);
</script></body></html>"""


class DebugWebServer:
    """Stores the latest telemetry/frame and serves them over localhost."""

    def __init__(self, host='127.0.0.1', port=9090, jpeg_quality=80,
                 stream_fps=8.0, config=None, config_path=None):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(np.clip(jpeg_quality, 30, 95))
        self.stream_interval = 1.0 / max(0.5, float(stream_fps))
        self._last_encode = 0.0
        self._condition = threading.Condition()
        self._jpeg = None
        self._sequence = 0
        self._status = {'state': 'starting', 'updated_at': time.time()}
        self._config = dict(config or {})
        self.config_path = config_path
        self._restart_after = None
        self._httpd = None
        self._thread = None
        self._running = False

    @property
    def url(self):
        visible_host = 'localhost' if self.host in ('127.0.0.1', '::1') else self.host
        return f'http://{visible_host}:{self.port}'

    @property
    def restart_requested(self):
        return (self._restart_after is not None and
                time.monotonic() >= self._restart_after)

    def start(self):
        self._httpd = ThreadingHTTPServer((self.host, self.port), _RequestHandler)
        self._httpd.daemon_threads = True
        self._httpd.dashboard = self
        self.port = self._httpd.server_address[1]
        self._running = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name='line-debug-web', daemon=True)
        self._thread.start()
        logger.info('网页调试已启动: %s', self.url)

    def stop(self):
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def update(self, frame, det, telemetry):
        status = dict(telemetry)
        status['updated_at'] = time.time()
        status['valid'] = bool(det.get('is_valid', False))
        status['point_count'] = len(det.get('points') or [])
        status['polarity'] = det.get('line_type', '')

        jpeg = None
        now = time.monotonic()
        if now - self._last_encode >= self.stream_interval:
            image = self._compose_debug_frame(frame, det)
            if image is not None:
                ok, encoded = cv2.imencode(
                    '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok:
                    jpeg = encoded.tobytes()
                    self._last_encode = now

        with self._condition:
            self._status = status
            if jpeg is not None:
                self._jpeg = jpeg
                self._sequence += 1
            self._condition.notify_all()

    def get_status(self):
        with self._condition:
            return dict(self._status)

    def get_config(self):
        with self._condition:
            return dict(self._config)

    def save_config(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError('请提交 JSON 对象')

        clean = {}
        for name, (value_type, minimum, maximum) in CONFIG_SCHEMA.items():
            if name not in submitted:
                raise ValueError(f'缺少参数: {name}')
            raw = submitted[name]
            if isinstance(raw, bool):
                raise ValueError(f'{name} 类型错误')
            try:
                value = value_type(raw)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f'{name} 不是有效数字') from None
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f'{name} 必须在 {minimum}~{maximum} 之间')
            clean[name] = value

        for name, choices in CONFIG_ENUMS.items():
            value = str(submitted.get(name, ''))
            if value not in choices:
                raise ValueError(f'{name} 必须是 {sorted(choices)} 之一')
            clean[name] = value

        # OpenCV 自适应阈值的邻域必须为奇数。
        if clean['adaptive_block'] % 2 == 0:
            clean['adaptive_block'] += 1
        if clean['adaptive_block'] > CONFIG_SCHEMA['adaptive_block'][2]:
            clean['adaptive_block'] -= 2

        if not self.config_path:
            raise ValueError('服务端未配置参数文件')
        temp_path = self.config_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(clean, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.config_path)

        with self._condition:
            self._config = clean
            # 留出时间让 HTTP 响应完整发回浏览器。
            self._restart_after = time.monotonic() + 0.5
        logger.info('网页参数已保存，准备安全重启')
        return clean

    def wait_for_frame(self, previous, timeout=2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous or not self._running,
                timeout=timeout)
            return self._sequence, self._jpeg, self._running

    @staticmethod
    def _compose_debug_frame(frame, det):
        if frame is None or getattr(frame, 'size', 0) == 0:
            return None
        work_width = 640
        h, w = frame.shape[:2]
        work_height = max(1, int(round(h * work_width / w)))
        raw = cv2.resize(frame, (work_width, work_height), interpolation=cv2.INTER_AREA)
        detector_width = max(1, int(det.get('work_width', 320)))
        scale = work_width / detector_width

        cv2.line(raw, (work_width // 2, 0), (work_width // 2, work_height),
                 (255, 100, 0), 1)
        for x, y, width in det.get('points') or []:
            cv2.circle(raw, (int(x * scale), int(y * scale)), 4, (40, 240, 90), -1)
        if det.get('is_valid') and det.get('points'):
            y0 = min(p[1] for p in det['points'])
            y1 = max(p[1] for p in det['points'])
            x0 = det['a'] * y0 + det['b']
            x1 = det['a'] * y1 + det['b']
            cv2.line(raw, (int(x0 * scale), int(y0 * scale)),
                     (int(x1 * scale), int(y1 * scale)), (0, 255, 255), 3)
        cv2.putText(raw, 'RAW + FIT', (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        binary_panel = np.full((work_height, work_width, 3), 255, np.uint8)
        binary = det.get('binary')
        if binary is not None and getattr(binary, 'size', 0):
            roi_top = max(0, int(round(det.get('roi_top', 0) * scale)))
            available = max(1, work_height - roi_top)
            binary_view = cv2.resize(cv2.bitwise_not(binary),
                                     (work_width, available),
                                     interpolation=cv2.INTER_NEAREST)
            binary_panel[roi_top:roi_top + available] = cv2.cvtColor(
                binary_view, cv2.COLOR_GRAY2BGR)
            cv2.line(binary_panel, (0, roi_top), (work_width, roi_top),
                     (0, 150, 255), 2)
        cv2.putText(binary_panel, 'BINARY ROI', (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 80, 220), 2)
        return np.hstack((raw, binary_panel))


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    @property
    def dashboard(self):
        return self.server.dashboard

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/':
            self._send_bytes(200, 'text/html; charset=utf-8',
                             _DASHBOARD_HTML.encode('utf-8'))
        elif path == '/api/status':
            payload = json.dumps(self.dashboard.get_status(), ensure_ascii=False,
                                 allow_nan=False).encode('utf-8')
            self._send_bytes(200, 'application/json; charset=utf-8', payload)
        elif path == '/api/config':
            payload = json.dumps(self.dashboard.get_config(), ensure_ascii=False,
                                 allow_nan=False).encode('utf-8')
            self._send_bytes(200, 'application/json; charset=utf-8', payload)
        elif path == '/stream.mjpg':
            self._stream_mjpeg()
        elif path == '/favicon.ico':
            self._send_bytes(204, 'image/x-icon', b'')
        else:
            self._send_bytes(404, 'text/plain; charset=utf-8', b'Not found')

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path != '/api/config':
            self._send_json(404, {'error': 'Not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 65536:
                raise ValueError('请求大小无效')
            submitted = json.loads(self.rfile.read(length).decode('utf-8'))
            config = self.dashboard.save_config(submitted)
            self._send_json(200, {'ok': True, 'config': config})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {'error': str(exc)})
        except OSError as exc:
            logger.exception('保存网页参数失败')
            self._send_json(500, {'error': f'保存失败: {exc}'})

    def _send_json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False,
                             allow_nan=False).encode('utf-8')
        self._send_bytes(status, 'application/json; charset=utf-8', payload)

    def _send_bytes(self, status, content_type, payload):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        sequence = -1
        try:
            while True:
                sequence, jpeg, running = self.dashboard.wait_for_frame(sequence)
                if not running:
                    break
                if jpeg is None:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii'))
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        logger.debug('HTTP %s - %s', self.address_string(), fmt % args)
