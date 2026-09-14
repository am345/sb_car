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
    'speed': (int, 0, 400),
    'max_z': (int, 0, 1500),
    'kp': (float, 0.0, 50.0),
    'kd': (float, 0.0, 20.0),
    'ka': (float, 0.0, 20.0),
    'err_alpha': (float, 0.01, 1.0),
    'z_rate': (float, 1.0, 500.0),
    'exposure': (int, 1, 5000),
    'roi_top': (float, 0.0, 0.9),
    'scan_start': (float, 0.0, 0.9),
    'crop_bottom': (float, 0.1, 1.0),
    'crop_top': (float, 0.1, 1.0),
    'track_half': (float, 5.0, 160.0),
    'startup_frames': (int, 1, 100),
    'ramp_frames': (int, 0, 200),
    'corner_delay_frames': (int, 0, 200),
    'corner_delay_speed': (int, 0, 300),
    'corner_delay_distance': (float, 0.0, 0.5),
    'corner_turn_degrees': (float, 10.0, 180.0),
    'corner_turn_speed': (int, 50, 1000),
    'lost_hold': (int, 0, 100),
    'search_frames': (int, 0, 200),
    'threshold': (int, 0, 255),
    'adaptive_block': (int, 3, 151),
    'adaptive_c': (float, -50.0, 50.0),
    'cross_lateral_distance_m': (float, 0.0, 2.0),
    'cross_lateral_speed': (int, 1, 300),
    'control_delay_m': (float, 0.0, 0.5),
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
    .chassis-control { display:flex; align-items:center; gap:16px; padding:14px 16px;
      margin-bottom:14px; flex-wrap:wrap; border-color:#36516c; }
    .chassis-copy { flex:1 1 320px; }
    .chassis-copy strong { display:block; font-size:18px; margin-bottom:2px; }
    .chassis-actions { display:flex; align-items:end; gap:10px; flex-wrap:wrap; }
    .chassis-speed { width:150px; }
    .chassis-speed label { display:block; color:var(--muted); font-size:14px; margin-bottom:4px; }
    button.chassis-start { background:var(--green); color:#04140b; min-width:120px; }
    button.chassis-stop { background:var(--yellow); color:#261b00; min-width:120px; }
    .manual-grid { display:grid; grid-template-columns:repeat(2,minmax(150px,240px)); gap:12px;
      align-items:end; }
    .manual-head { display:flex; align-items:center; justify-content:space-between; gap:14px; }
    button.manual-on { background:var(--yellow); color:#241800; }
    button.cancel { background:#607086; color:white; }
    button { border:0; border-radius:9px; padding:10px 18px; background:var(--cyan); color:#041016;
      font-weight:700; cursor:pointer; }
    button:disabled { opacity:.5; cursor:wait; }
    button.emergency { background:var(--red); color:white; font-size:16px; padding:11px 22px;
      box-shadow:0 0 0 2px rgba(255,101,119,.22); }
    button.emergency:active { transform:translateY(1px); background:#e43f55; }
    #saveResult { color:var(--muted); }
    footer { color:var(--muted); margin-top:10px; text-align:right; }
    @media (max-width:850px) { .layout { grid-template-columns:1fr; } .form-grid { grid-template-columns:repeat(2,1fr); } }
    @media (max-width:520px) { .form-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <header><div><h1>视觉循迹实时调试</h1><div class="sub">RAW + 拟合结果 / BINARY 检测区</div></div>
    <div style="display:flex;align-items:center;gap:12px"><button id="emergencyButton" class="emergency" type="button">急停</button>
      <a href="/logs" style="color:var(--cyan)">图像日志</a><span id="connection" class="warn">正在连接…</span></div></header>
  <section id="chassisControl" class="card chassis-control">
    <div class="chassis-copy"><strong id="chassisStatus">底盘状态读取中…</strong>
      <span class="config-note" id="chassisHint">程序启动后默认保持停用。</span>
      <div id="chassisResult" class="bad" role="status" aria-live="polite"></div></div>
    <div class="chassis-actions">
      <div class="chassis-speed"><label for="chassisSpeed">启动速度 mm/s</label>
        <input id="chassisSpeed" type="number" min="1" max="400" step="1" value="100"></div>
      <button id="chassisToggle" class="chassis-start" type="button" disabled>启动底盘</button>
    </div>
  </section>
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
        <div class="metric"><div class="label">底盘电压</div><div id="batteryVoltage" class="value">--</div></div>
        <div class="metric"><div class="label">仿真实测速度</div><div id="simSpeed" class="value">--</div></div>
        <div class="metric"><div class="label">仿真实时倍率</div><div id="simRate" class="value">--</div></div>
        <div class="metric"><div class="label">转向速度 mrad/s</div><div id="turn" class="value">--</div></div>
      </div>
      <div class="bars">
        <div><div class="bar-label"><span>转向负载</span><span id="turnPct">--</span></div><div class="track"><div id="turnBar" class="fill"></div></div></div>
        <div><div class="bar-label"><span>起步确认</span><span id="startCount">--</span></div><div class="track"><div id="startBar" class="fill"></div></div></div>
        <div class="metric"><div class="label">检测点 / 失线 / 相机无帧</div><div id="counts" class="value small">--</div></div>
        <div class="metric"><div class="label">P / D / 角度前馈</div><div id="terms" class="value small">--</div></div>
        <div class="metric"><div class="label">二值化 / 线极性</div><div id="mode" class="value small">--</div></div>
        <div class="metric wide"><div class="label">IMU / 速度里程计</div><div id="odomPose" class="value small">X -- / Y -- / 航向 --</div></div>
        <div class="metric"><div class="label">累计里程</div><div id="odomDistance" class="value small">-- m</div></div>
        <button id="resetOdomButton" type="button">里程清零</button>
      </div>
    </aside>
  </section>
  <section id="trafficPanel" class="card config">
    <h2>交通标志识别</h2>
    <div id="trafficModeNote" class="config-note">仅显示置信度 &gt;60% 的识别结果，连续 3 帧确认。</div>
    <div id="trafficBehavior" class="config-note"></div>
    <div id="trafficCooldown" class="config-note"></div>
    <div id="trafficState">未启用</div>
    <div id="trafficRate" class="config-note"></div>
    <div id="trafficResults"></div>
  </section>
  <section class="card config">
    <div class="manual-head"><div><h2>手动里程控制</h2>
      <div class="config-note">正距离=前进，负距离=后退；正角度=左转，负角度=右转。先转向，再按目标航向行驶。</div></div>
      <button id="manualToggle" type="button">打开手动控制</button></div>
    <div class="manual-grid">
      <div class="field"><label for="targetDistance">相对距离 m（-5～5）</label><input id="targetDistance" type="number" min="-5" max="5" step="0.05" value="0"></div>
      <div class="field"><label for="targetAngle">相对角度 °（左正右负）</label><input id="targetAngle" type="number" min="-360" max="360" step="1" value="0"></div>
    </div>
    <div class="actions"><button id="runTargetButton" type="button" disabled>执行目标</button>
      <button id="cancelTargetButton" class="cancel" type="button" disabled>取消并停车</button>
      <span id="manualResult">视觉循迹模式</span></div>
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
let manualMode=false, chassisArmed=false, chassisBusy=false;
const fields=[
 ['speed','巡航速度 mm/s','number','1'],['max_z','最大转向 mrad/s','number','1'],
 ['kp','P 增益','number','0.1'],['kd','D 增益','number','0.1'],['ka','角度前馈','number','0.1'],
 ['err_alpha','滤波系数','number','0.05'],['z_rate','转向变化限制','number','1'],
 ['exposure','摄像头手动曝光','number','1'],
 ['roi_top','ROI 起点','number','0.05'],['scan_start','扫描起点','number','0.05'],
 ['crop_bottom','近处宽度比','number','0.05'],['crop_top','远处宽度比','number','0.05'],
 ['track_half','搜索窗半宽 px','number','1'],['startup_frames','起步确认帧','number','1'],
 ['ramp_frames','加速斜坡帧','number','1'],['corner_delay_frames','L弯转向延迟帧','number','1'],
 ['corner_delay_speed','L弯延迟速度 mm/s','number','1'],
 ['corner_delay_distance','L弯延迟距离 m','number','0.01'],
 ['corner_turn_degrees','L弯旋转角度 °','number','1'],
 ['corner_turn_speed','L弯旋转速度 mrad/s','number','1'],
 ['lost_hold','失线保持帧','number','1'],
 ['search_frames','失线搜索帧','number','1'],['threshold','固定阈值','number','1'],
 ['adaptive_block','自适应邻域','number','2'],['adaptive_c','自适应 C','number','0.5'],
 ['cross_lateral_distance_m','cross左移距离 m','number','0.01'],
 ['cross_lateral_speed','cross横移速度 mm/s','number','1'],
 ['control_delay_m','路径延迟距离 m','number','0.01'],
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
async function loadConfig(){
 const r=await fetch('/api/config',{cache:'no-store'}), c=await r.json(); buildForm(c);
 const configured=Number(c.speed); $('chassisSpeed').value=configured>0?configured:100;
}
function refreshTraffic(t){
  t=t||{state:'disabled'};
  const states={disabled:'未启用',loading:'模型加载中…',ready:'识别中',stale:'画面已过期，等待新帧',error:'识别不可用'};
  $('trafficState').textContent=(states[t.state]||t.state)+(t.error?`：${t.error}`:'')+
    (t.state==='ready'?(t.confirmed?` · 已确认 ${t.confirmed}`:` · 确认 ${t.confirm_count||0}/3`):'');
  $('trafficRate').textContent=t.state==='ready'?`${t.backend||''} · 推理 ${num(t.inference_ms)} ms（${num(t.inference_fps)} FPS） · 结果更新 ${num(t.result_fps)} FPS · 数据年龄 ${num(t.age_sec)} s`:'';
  const fresh=t.state==='ready';
  $('trafficResults').replaceChildren();
  if(fresh){
    if(!t.detections?.length) $('trafficResults').textContent='当前未检测到标志';
    for(const d of t.detections||[]){
      const row=document.createElement('div'); row.textContent=`${d.label} · ${(d.confidence*100).toFixed(1)}% · 有效检测`;
      $('trafficResults').appendChild(row);
    }
  }
}
async function refresh(){
  try {
    const r=await fetch('/api/status',{cache:'no-store'}); if(!r.ok) throw Error(r.status);
    const d=await r.json(), age=Date.now()/1000-d.updated_at;
    const simulation=Boolean(d.simulation_mode);
    if(simulation) document.querySelector('h1').textContent='Webots 视觉循迹仿真';
    refreshTraffic(d.traffic);
    chassisArmed=Boolean(d.chassis_armed);
    const chassisAvailable=Boolean(simulation||(d.traffic_control&&d.chassis_connected));
    $('chassisStatus').textContent=simulation?'Webots 虚拟麦轮底盘':(!chassisAvailable?'底盘控制未连接':
      (chassisArmed?'底盘已启动':'底盘已停用'));
    $('chassisStatus').className=chassisArmed?'ok':(d.chassis_fault?'bad':'warn');
    $('chassisHint').textContent=d.chassis_fault?`保护停车：${d.chassis_fault}`:
      (simulation?(chassisArmed?`仿真运行中，目标速度 ${d.chassis_target_speed||0} mm/s。`:
       (d.valid?'线路有效；设置速度后可启动虚拟底盘。':'当前未检测到有效黑线。')):
       (chassisArmed?`运动已授权，速度上限 ${d.chassis_target_speed||0} mm/s；关闭页面超过3秒将自动停车。`:
        (chassisAvailable?(d.valid?'线路有效，启动前将检查模型、底盘反馈和网页心跳。':'未检测到有效线路：请将摄像头对准赛道，让黑线进入紫色ROI区域。'):'当前为仅视觉模式，不能启动底盘。')));
    $('chassisToggle').textContent=chassisArmed?'停用底盘':'启动底盘';
    $('chassisToggle').className=chassisArmed?'chassis-stop':'chassis-start';
    $('chassisToggle').disabled=chassisBusy||!chassisAvailable;
    $('chassisSpeed').disabled=chassisArmed||!chassisAvailable;
  $('trafficModeNote').textContent=d.traffic_control?'交通控制已启用；只接收置信度 >60% 的结果，各类别独立三帧确认，停车优先。':'仅显示置信度 >60% 的识别结果，不控制车辆；连续3帧确认。';
  if(d.sign_only) $('trafficModeNote').textContent='纯标志控制：只接收置信度 >60% 的结果；确认后立即动作。环岛仅显示，不执行。无标志时按当前巡航目标直行。';
    const b=d.behavior;
    $('trafficBehavior').textContent=b?`行为 ${b.state} · 任务 ${b.task||'无'} · 锁定支路 ${b.locked_branch||'无'} · 巡航目标 ${b.cruise_mm_s} / 硬件上限 ${b.speed_ceiling_mm_s} mm/s · 横移 ${d.lateral_speed||0} mm/s${b.fault?' · 保护停车：'+b.fault:''}`:'';
    $('trafficCooldown').textContent=Object.entries(b?.sign_locks||{}).map(([label,s])=>{
      const phase=s.phase==='executing'?'执行中':s.phase==='cooldown'?`冷却 ${num(s.remaining_sec)}秒`:`等待标志消失（还需 ${num(s.clear_remaining_sec)}秒）`;
      const outcome=s.outcome==='interrupted'?' · 已中断':s.outcome==='ignored'?' · 本轮忽略':'';
      return `${label}：${phase}${outcome}`;
    }).join('；');
    if(d.traffic_control) await fetch('/api/heartbeat',{method:'POST',cache:'no-store'});
    if(d.vision_only && $('trafficPanel').nextElementSibling!==document.querySelector('.layout')){
      document.querySelector('.layout').before($('trafficPanel'));
      document.querySelector('h1').textContent='交通标志实时识别';
    }
    $('connection').textContent=age<2?'已连接':`数据延迟 ${num(age)}s`;
    $('connection').className=age<2?'ok':'warn';
    $('state').textContent=d.state||'等待数据';
    $('valid').textContent=d.valid?'有效':'无效'; $('valid').className='value '+(d.valid?'ok':'bad');
    $('fps').textContent=num(d.fps); $('error').textContent=`${num(d.error_px)} px`;
    $('angle').textContent=`${num(d.angle_deg)}°`; $('speed').textContent=`${num(d.speed,0)} mm/s`;
    const voltage=Number(d.battery_voltage), hasVoltage=Number.isFinite(voltage);
    $('batteryVoltage').textContent=hasVoltage?`${voltage.toFixed(2)} V`:'--';
    $('batteryVoltage').className='value '+(!hasVoltage?'':voltage<10.5?'bad':voltage<11?'warn':'ok');
    $('simSpeed').textContent=simulation?`${num(d.sim_actual_speed_mm_s,0)} mm/s`:'--';
    $('simRate').textContent=simulation?`${num(d.sim_realtime_factor,2)}×`:'--';
    $('turn').textContent=`${num(d.turn,0)} mrad/s`; $('frame').textContent=`frame ${d.frame_count??'--'}`;
    const tp=Math.min(100,Math.abs(d.turn||0)/Math.max(1,d.max_z||1)*100);
    $('turnPct').textContent=`${num(tp,0)}%`; $('turnBar').style.width=tp+'%';
    const sp=d.started?100:Math.min(100,(d.start_seen||0)/Math.max(1,d.startup_frames||1)*100);
    $('startBar').style.width=sp+'%'; $('startCount').textContent=d.started?'已起步':`${d.start_seen||0}/${d.startup_frames||0}`;
    $('counts').textContent=`${d.point_count||0} / ${d.lost_count||0} / ${d.no_frame_count||0}`;
    $('terms').textContent=`${num(d.p_term)} / ${num(d.d_term)} / ${num(d.angle_term)}`;
    $('mode').textContent=`${d.binary_mode||'--'} / ${d.polarity||'--'}`;
    $('odomPose').textContent=`X ${num(d.odom_x_m,3)} m / Y ${num(d.odom_y_m,3)} m / 航向 ${num(d.odom_yaw_deg)}°`;
    $('odomDistance').textContent=`${num(d.odom_distance_m,3)} m`;
    manualMode=Boolean(d.manual_mode);
    $('manualToggle').textContent=manualMode?'关闭手动控制':'打开手动控制';
    $('manualToggle').className=manualMode?'manual-on':'';
    $('targetDistance').disabled=!manualMode; $('targetAngle').disabled=!manualMode;
    $('runTargetButton').disabled=!manualMode;
    $('cancelTargetButton').disabled=!manualMode;
    $('manualToggle').disabled=Boolean(d.vision_only||d.traffic_control||simulation);
    $('resetOdomButton').disabled=Boolean(d.vision_only||d.traffic_control);
    $('saveButton').disabled=Boolean(d.vision_only||simulation);
    if(d.vision_only) $('saveResult').textContent='仅视觉预览，运行参数只读';
    if(simulation) $('saveResult').textContent='仿真模式：参数暂为只读';
    const remain=d.manual_phase==='rotate'?`，剩余 ${num(d.manual_remaining_deg)}°`:
      (d.manual_phase==='drive'?`，剩余 ${num(d.manual_remaining_m,3)} m`:'');
    $('manualResult').textContent=(d.manual_state||'视觉循迹')+remain;
  } catch(e) { $('connection').textContent='连接中断'; $('connection').className='bad'; refreshTraffic({state:'stale'}); }
}
refresh(); setInterval(refresh,200);
$('chassisToggle').addEventListener('click',async()=>{
 const enabling=!chassisArmed, speed=Number($('chassisSpeed').value);
 if(enabling&&(!Number.isInteger(speed)||speed<1||speed>400)){
  $('chassisResult').textContent='启动速度必须是1~400 mm/s的整数。'; return;
 }
 if(enabling&&!confirm(`确认启动底盘？\n\n车辆将在安全检查通过并连续检测到线路后，以最高 ${speed} mm/s 起步。请确保车辆位于赛道、周围无人。`)) return;
 chassisBusy=true; $('chassisToggle').disabled=true; $('chassisResult').textContent='';
 $('chassisHint').textContent=enabling?'正在执行启动前安全检查…':'正在停车…';
 try {
  const body={armed:enabling,speed,confirmation:enabling?'ENABLE_CHASSIS':''};
  const r=await fetch('/api/chassis/arm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status);
  chassisArmed=Boolean(answer.chassis_armed); $('chassisHint').textContent=answer.chassis_state;
 } catch(err) { $('chassisResult').textContent='操作失败：'+err.message; }
 finally { chassisBusy=false; }
});
$('emergencyButton').addEventListener('click',async()=>{
 const button=$('emergencyButton'); button.disabled=true; button.textContent='停车中…';
 $('connection').textContent='正在急停'; $('connection').className='bad';
 try {
  const r=await fetch('/api/emergency-stop',{method:'POST',cache:'no-store',keepalive:true});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status);
  button.textContent='已急停'; $('connection').textContent='车辆已停车';
 } catch(err) {
  button.disabled=false; button.textContent='急停';
  $('connection').textContent='急停请求失败：'+err.message;
 }
});
$('resetOdomButton').addEventListener('click',async()=>{
 const button=$('resetOdomButton'); button.disabled=true; button.textContent='清零中…';
 try {
  const r=await fetch('/api/odometry/reset',{method:'POST',cache:'no-store'});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status);
  button.textContent='已清零'; setTimeout(()=>{button.textContent='里程清零';button.disabled=false;},800);
 } catch(err) { button.textContent='清零失败'; button.disabled=false; }
});
$('manualToggle').addEventListener('click',async()=>{
 const button=$('manualToggle'); button.disabled=true;
 try {
  const r=await fetch('/api/control/mode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({manual:!manualMode})});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status);
  manualMode=Boolean(answer.manual_mode); $('manualResult').textContent=answer.manual_state;
 } catch(err) { $('manualResult').textContent='切换失败：'+err.message; }
 finally { button.disabled=false; }
});
$('runTargetButton').addEventListener('click',async()=>{
 const button=$('runTargetButton'); button.disabled=true; $('manualResult').textContent='正在下发目标…';
 try {
  const body={distance_m:Number($('targetDistance').value),angle_deg:Number($('targetAngle').value)};
  const r=await fetch('/api/control/target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const answer=await r.json(); if(!r.ok) throw Error(answer.error||r.status);
  $('manualResult').textContent=answer.manual_state;
 } catch(err) { $('manualResult').textContent='执行失败：'+err.message; }
 finally { button.disabled=!manualMode; }
});
$('cancelTargetButton').addEventListener('click',async()=>{
 try {
  const r=await fetch('/api/control/cancel',{method:'POST'}); const answer=await r.json();
  if(!r.ok) throw Error(answer.error||r.status); $('manualResult').textContent=answer.manual_state;
 } catch(err) { $('manualResult').textContent='停车失败：'+err.message; }
});
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


_LOGS_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>图像日志</title><style>
body{margin:0;background:#07101c;color:#e7edf5;font:14px system-ui,sans-serif}main{width:min(1500px,96vw);margin:20px auto}
a{color:#54d6ff}.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.item{border:1px solid #26364c;border-radius:10px;overflow:hidden;background:#101a29}.item img{display:block;width:100%}.name{padding:8px 10px;color:#a8b3c2;font-family:monospace}
</style></head><body><main><div class="head"><div><h1>图像日志</h1><div>每秒一张，最多保留最近 300 张</div></div><a href="/">返回实时页面</a></div><div id="grid" class="grid">加载中…</div></main>
<script>
async function refresh(){const r=await fetch('/api/image-logs',{cache:'no-store'}),d=await r.json(),g=document.getElementById('grid');g.innerHTML='';
for(const x of d.logs){const box=document.createElement('div');box.className='item';const a=document.createElement('a');a.href=x.url;a.target='_blank';const img=document.createElement('img');img.src=x.url;img.loading='lazy';a.appendChild(img);const name=document.createElement('div');name.className='name';name.textContent=x.name;box.append(a,name);g.appendChild(box)}if(!d.logs.length)g.textContent='暂无图像日志';}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class DebugWebServer:
    """Stores the latest telemetry/frame and serves them over localhost."""

    def __init__(self, host='127.0.0.1', port=9090, jpeg_quality=80,
                 stream_fps=8.0, config=None, config_path=None,
                 emergency_callback=None, reset_odometry_callback=None,
                 manual_mode_callback=None, manual_target_callback=None,
                 manual_cancel_callback=None, traffic_model=None,
                 chassis_arm_callback=None):
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
        self.image_log_dir = (os.path.join(os.path.dirname(config_path), 'image_logs')
                              if config_path else None)
        self.image_log_interval = 1.0
        self.image_log_limit = 300
        self._last_image_log = 0.0
        self._restart_after = None
        self._emergency_callback = emergency_callback
        self._reset_odometry_callback = reset_odometry_callback
        self._manual_mode_callback = manual_mode_callback
        self._manual_target_callback = manual_target_callback
        self._manual_cancel_callback = manual_cancel_callback
        self._chassis_arm_callback = chassis_arm_callback
        self._httpd = None
        self._thread = None
        self._render_thread = None
        self._render_job = None
        self._running = False
        self._traffic = None
        self._heartbeat_at = None
        if traffic_model:
            from core.traffic_signs import TrafficSignWorker
            self._traffic = TrafficSignWorker(traffic_model)

    @property
    def url(self):
        visible_host = 'localhost' if self.host in ('127.0.0.1', '::1') else self.host
        return f'http://{visible_host}:{self.port}'

    @property
    def restart_requested(self):
        return (self._restart_after is not None and
                time.monotonic() >= self._restart_after)

    def start(self):
        if self.image_log_dir:
            try:
                os.makedirs(self.image_log_dir, exist_ok=True)
            except OSError as exc:
                logger.warning('无法创建图像日志目录 %s: %s', self.image_log_dir, exc)
                self.image_log_dir = None
        self._httpd = ThreadingHTTPServer((self.host, self.port), _RequestHandler)
        self._httpd.daemon_threads = True
        self._httpd.dashboard = self
        self.port = self._httpd.server_address[1]
        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop, name='line-debug-render', daemon=True)
        self._render_thread.start()
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name='line-debug-web', daemon=True)
        self._thread.start()
        if self._traffic is not None:
            self._traffic.start()
        logger.info('网页调试已启动: %s', self.url)

    def stop(self):
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None
        if self._traffic is not None:
            self._traffic.stop()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def update(self, frame, det, telemetry):
        if self._traffic is not None:
            self._traffic.submit(frame)
        status = dict(telemetry)
        status['updated_at'] = time.time()
        status['valid'] = bool(det.get('is_valid', False))
        status['point_count'] = len(det.get('points') or [])
        status['polarity'] = det.get('line_type', '')
        status['corner_dir'] = int(det.get('corner_dir', 0))
        status['corner_y_ratio'] = float(det.get('corner_y_ratio', 0.0))
        status['corner_span'] = float(det.get('corner_span', 0.0))
        status['junction_straight'] = bool(det.get('junction_straight', False))

        now = time.monotonic()
        with self._condition:
            self._status = status
            if (self._running and frame is not None and
                    getattr(frame, 'size', 0) and
                    now - self._last_encode >= self.stream_interval):
                # The control loop only publishes the latest render job.  All
                # expensive drawing, JPEG encoding and log I/O stay off its
                # timing-critical thread.
                self._render_job = (frame.copy(), dict(det), dict(status))
                self._last_encode = now
            self._condition.notify_all()

    def _render_loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._render_job is not None or not self._running)
                if not self._running:
                    return
                frame, det, status = self._render_job
                self._render_job = None

            image = self._compose_debug_frame(frame, det)
            if image is None:
                continue
            ok, encoded = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue
            jpeg = encoded.tobytes()
            now = time.monotonic()
            if (self.image_log_dir and
                    now - self._last_image_log >= self.image_log_interval):
                self._save_image_log(jpeg, status)
                self._last_image_log = now

            with self._condition:
                self._jpeg = jpeg
                self._sequence += 1
                self._condition.notify_all()

    def get_status(self):
        with self._condition:
            status = dict(self._status)
        status['traffic'] = self.get_traffic_status()
        status['traffic']['control_enabled'] = bool(status.get('traffic_control', False))
        return status

    def get_traffic_status(self):
        return self._traffic.snapshot() if self._traffic else {'state': 'disabled'}

    def heartbeat(self):
        with self._condition:
            self._heartbeat_at = time.monotonic()

    def heartbeat_age(self):
        with self._condition:
            return 1e9 if self._heartbeat_at is None else time.monotonic()-self._heartbeat_at

    def get_traffic_image(self):
        return self._traffic.jpeg() if self._traffic else None

    def wait_for_traffic_frame(self, previous):
        if self._traffic is None:
            return previous, None, False
        return self._traffic.wait_for_frame(previous)

    def get_config(self):
        with self._condition:
            return dict(self._config)

    def emergency_stop(self):
        """立即调用控制层停车；不经过参数保存或重启路径。"""
        if self._emergency_callback is None:
            raise RuntimeError('急停通道未配置')
        self._restart_after = None
        self._emergency_callback()
        with self._condition:
            self._status['state'] = 'emergency-stop'
            self._status['speed'] = 0
            self._status['lateral_speed'] = 0
            self._status['turn'] = 0
            self._status['updated_at'] = time.time()
            self._condition.notify_all()
        logger.warning('网页急停已执行')

    def reset_odometry(self):
        if self._reset_odometry_callback is None:
            raise RuntimeError('里程计清零通道未配置')
        self._reset_odometry_callback()
        with self._condition:
            self._status.update({
                'odom_x_m': 0.0, 'odom_y_m': 0.0,
                'odom_yaw_deg': 0.0, 'odom_distance_m': 0.0,
            })
            self._condition.notify_all()

    def set_manual_mode(self, enabled):
        if self._manual_mode_callback is None:
            raise RuntimeError('手动控制通道未配置')
        return self._manual_mode_callback(bool(enabled))

    def start_manual_target(self, distance_m, angle_deg):
        if self._manual_target_callback is None:
            raise RuntimeError('里程目标通道未配置')
        return self._manual_target_callback(distance_m, angle_deg)

    def cancel_manual_target(self):
        if self._manual_cancel_callback is None:
            raise RuntimeError('手动停车通道未配置')
        return self._manual_cancel_callback()

    def set_chassis_armed(self, enabled, speed=None, confirmation=None):
        if self._chassis_arm_callback is None:
            raise RuntimeError('底盘启停通道未配置')
        if enabled and confirmation != 'ENABLE_CHASSIS':
            raise ValueError('启动底盘需要明确确认')
        return self._chassis_arm_callback(bool(enabled), speed)

    def get_image_logs(self):
        if not self.image_log_dir:
            return []
        try:
            names = sorted(
                (entry.name for entry in os.scandir(self.image_log_dir)
                 if entry.is_file() and entry.name.endswith('.jpg')),
                reverse=True)
        except OSError:
            return []
        return [{'name': name, 'url': '/image-logs/' + name}
                for name in names[:self.image_log_limit]]

    def read_image_log(self, name):
        if (not self.image_log_dir or os.path.basename(name) != name or
                not name.endswith('.jpg')):
            return None
        try:
            with open(os.path.join(self.image_log_dir, name), 'rb') as stream:
                return stream.read()
        except OSError:
            return None

    def _save_image_log(self, jpeg, status):
        state = ''.join(c if c.isalnum() or c in '-_' else '-'
                        for c in str(status.get('state', 'unknown')))
        stamp = time.strftime('%Y%m%d_%H%M%S')
        frame_count = int(status.get('frame_count', 0))
        name = f'{stamp}_{state}_f{frame_count:08d}.jpg'
        try:
            with open(os.path.join(self.image_log_dir, name), 'wb') as stream:
                stream.write(jpeg)
            entries = sorted(
                (entry for entry in os.scandir(self.image_log_dir)
                 if entry.is_file() and entry.name.endswith('.jpg')),
                key=lambda entry: entry.name, reverse=True)
            for entry in entries[self.image_log_limit:]:
                os.unlink(entry.path)
        except OSError as exc:
            logger.warning('写入图像日志失败，已停用: %s', exc)
            self.image_log_dir = None

    def save_config(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError('请提交 JSON 对象')

        clean = {}
        for name, (value_type, minimum, maximum) in CONFIG_SCHEMA.items():
            if name not in submitted:
                # A dashboard tab may remain open across a deployment that
                # adds a new setting. Preserve the server-side current value
                # instead of rejecting every older form submission.
                if name not in self._config:
                    raise ValueError(f'缺少参数: {name}')
                raw = self._config[name]
            else:
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

        # 检测器使用的是以画面中心为轴、上宽下窄的梯形 ROI。
        # 在原图和二值图上画出同一组边界，便于直接判断线路是否被裁掉。
        roi_top = max(0, min(work_height - 1,
                             int(round(det.get('roi_top', 0) * scale))))
        top_frac = float(np.clip(det.get('crop_top_frac', 1.0), 0.0, 1.0))
        bottom_frac = float(np.clip(det.get('crop_bottom_frac', 1.0), 0.0, 1.0))
        center = work_width / 2.0
        top_half = top_frac * work_width * 0.5
        bottom_half = bottom_frac * work_width * 0.5
        roi_outline = np.array([
            [int(round(center - top_half)), roi_top],
            [int(round(center + top_half)) - 1, roi_top],
            [int(round(center + bottom_half)) - 1, work_height - 1],
            [int(round(center - bottom_half)), work_height - 1],
        ], dtype=np.int32)

        cv2.line(raw, (work_width // 2, 0), (work_width // 2, work_height),
                 (255, 100, 0), 1)
        cv2.polylines(raw, [roi_outline], True, (255, 0, 255), 2,
                      lineType=cv2.LINE_AA)
        cv2.putText(raw, 'ROI', (roi_outline[0, 0] + 6, roi_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        branch_colors = {
            'left': (255, 120, 0),
            'straight': (255, 255, 0),
            'right': (0, 120, 255),
        }
        blocked_branches = set(det.get('blocked_branches') or [])
        for candidate in det.get('branch_candidates') or []:
            coeffs = candidate.get('fit_coeffs')
            candidate_points = candidate.get('points') or []
            direction = candidate.get('direction')
            if len(candidate_points) < 3:
                continue
            if coeffs is not None and len(coeffs) >= 2:
                y0 = min(point[1] for point in candidate_points)
                y1 = max(point[1] for point in candidate_points)
                fit_ys = np.linspace(y0, y1, 40)
                fit_xs = np.polyval(coeffs, fit_ys)
                curve = np.column_stack((fit_xs * scale, fit_ys * scale))
            else:
                curve = np.asarray([[point[0]*scale, point[1]*scale]
                                    for point in candidate_points],
                                   dtype=np.float64)
            curve[:, 0] = np.clip(curve[:, 0], 0, work_width - 1)
            curve[:, 1] = np.clip(curve[:, 1], 0, work_height - 1)
            blocked = direction in blocked_branches
            color = ((0, 0, 255) if blocked else
                     branch_colors.get(direction, (180, 180, 180)))
            cv2.polylines(raw, [np.rint(curve).astype(np.int32)], False,
                          color, 2, lineType=cv2.LINE_AA)
            label_at = tuple(np.rint(curve[0]).astype(int))
            label = (f'BLOCKED-{str(direction).upper()}' if blocked
                     else str(direction).upper())
            cv2.putText(raw, label, label_at,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
                        lineType=cv2.LINE_AA)
        for x, y, width in det.get('points') or []:
            cv2.circle(raw, (int(x * scale), int(y * scale)), 4, (40, 240, 90), -1)
        if det.get('is_valid') and det.get('points'):
            fit_range = det.get('fit_y_range')
            y0 = (fit_range[0] if fit_range is not None else
                  min(p[1] for p in det['points']))
            y1 = (fit_range[1] if fit_range is not None else
                  max(p[1] for p in det['points']))
            coeffs = det.get('fit_coeffs')
            if coeffs is not None and len(coeffs) == 3:
                fit_ys = np.linspace(y0, y1, 40)
                fit_xs = np.polyval(coeffs, fit_ys)
                curve = np.column_stack((fit_xs * scale, fit_ys * scale))
                curve[:, 0] = np.clip(curve[:, 0], 0, work_width - 1)
                curve[:, 1] = np.clip(curve[:, 1], 0, work_height - 1)
                cv2.polylines(raw, [np.rint(curve).astype(np.int32)], False,
                              (0, 255, 255), 3, lineType=cv2.LINE_AA)
        corner = det.get('corner_point')
        corner_dir = int(det.get('corner_dir', 0))
        detailed_branches = any(candidate.get('points')
                                for candidate in
                                (det.get('branch_candidates') or []))
        if (corner is not None and det.get('junction_straight') and
                not detailed_branches):
            junction_px = (int(round(corner[0] * scale)),
                           int(round(corner[1] * scale)))
            cv2.circle(raw, junction_px, 9, (255, 255, 0), 2,
                       lineType=cv2.LINE_AA)
            cv2.putText(raw, 'CROSS-STRAIGHT',
                        (max(5, junction_px[0] - 75), max(24, junction_px[1] - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if corner is not None and corner_dir:
            corner_px = (int(round(corner[0] * scale)),
                         int(round(corner[1] * scale)))
            arrow_px = (corner_px[0] + corner_dir * 70, corner_px[1])
            cv2.circle(raw, corner_px, 8, (0, 80, 255), 2,
                       lineType=cv2.LINE_AA)
            cv2.arrowedLine(raw, corner_px, arrow_px, (0, 80, 255), 3,
                            line_type=cv2.LINE_AA, tipLength=0.25)
            cv2.putText(raw, 'L-RIGHT' if corner_dir > 0 else 'L-LEFT',
                        (max(5, corner_px[0] - 45), max(24, corner_px[1] - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
        cv2.putText(raw, 'RAW + FIT', (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        binary_panel = np.full((work_height, work_width, 3), 255, np.uint8)
        binary = det.get('binary')
        if binary is not None and getattr(binary, 'size', 0):
            available = max(1, work_height - roi_top)
            binary_view = cv2.resize(cv2.bitwise_not(binary),
                                     (work_width, available),
                                     interpolation=cv2.INTER_NEAREST)
            binary_panel[roi_top:roi_top + available] = cv2.cvtColor(
                binary_view, cv2.COLOR_GRAY2BGR)
            cv2.line(binary_panel, (0, roi_top), (work_width, roi_top),
                     (0, 150, 255), 2)
        cv2.polylines(binary_panel, [roi_outline], True, (255, 0, 255), 2,
                      lineType=cv2.LINE_AA)
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
        elif path == '/logs':
            self._send_bytes(200, 'text/html; charset=utf-8',
                             _LOGS_HTML.encode('utf-8'))
        elif path == '/api/status':
            payload = json.dumps(self.dashboard.get_status(), ensure_ascii=False,
                                 allow_nan=False).encode('utf-8')
            self._send_bytes(200, 'application/json; charset=utf-8', payload)
        elif path == '/api/config':
            payload = json.dumps(self.dashboard.get_config(), ensure_ascii=False,
                                 allow_nan=False).encode('utf-8')
            self._send_bytes(200, 'application/json; charset=utf-8', payload)
        elif path == '/api/traffic-image':
            payload = self.dashboard.get_traffic_image()
            self._send_bytes(200 if payload else 404, 'image/jpeg', payload or b'')
        elif path == '/api/image-logs':
            payload = json.dumps({'logs': self.dashboard.get_image_logs()},
                                 ensure_ascii=False).encode('utf-8')
            self._send_bytes(200, 'application/json; charset=utf-8', payload)
        elif path.startswith('/image-logs/'):
            payload = self.dashboard.read_image_log(path[len('/image-logs/'):])
            if payload is None:
                self._send_bytes(404, 'text/plain; charset=utf-8', b'Not found')
            else:
                self._send_bytes(200, 'image/jpeg', payload)
        elif path == '/stream.mjpg':
            self._stream_mjpeg()
        elif path == '/traffic.mjpg':
            self._stream_mjpeg(traffic=True)
        elif path == '/favicon.ico':
            self._send_bytes(204, 'image/x-icon', b'')
        else:
            self._send_bytes(404, 'text/plain; charset=utf-8', b'Not found')

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/heartbeat':
            self.dashboard.heartbeat()
            self._send_json(200, {'ok': True})
            return
        if path == '/api/emergency-stop':
            try:
                self.dashboard.emergency_stop()
                self._send_json(200, {'ok': True, 'state': 'emergency-stop'})
            except Exception as exc:
                logger.exception('网页急停失败')
                self._send_json(500, {'ok': False, 'error': str(exc)})
            return
        if path == '/api/odometry/reset':
            try:
                self.dashboard.reset_odometry()
                self._send_json(200, {'ok': True})
            except Exception as exc:
                logger.exception('里程计清零失败')
                self._send_json(500, {'ok': False, 'error': str(exc)})
            return
        if path == '/api/chassis/arm':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 4096:
                    raise ValueError('请求大小无效')
                submitted = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(submitted.get('armed'), bool):
                    raise ValueError('armed 必须为布尔值')
                result = self.dashboard.set_chassis_armed(
                    submitted['armed'], submitted.get('speed'),
                    submitted.get('confirmation'))
                self._send_json(200, {'ok': True, **result})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {'ok': False, 'error': str(exc)})
            except Exception as exc:
                logger.exception('网页底盘启停失败')
                self._send_json(409, {'ok': False, 'error': str(exc)})
            return
        if path in ('/api/control/mode', '/api/control/target',
                    '/api/control/cancel'):
            try:
                if path == '/api/control/cancel':
                    result = self.dashboard.cancel_manual_target()
                else:
                    length = int(self.headers.get('Content-Length', '0'))
                    if length <= 0 or length > 4096:
                        raise ValueError('请求大小无效')
                    submitted = json.loads(
                        self.rfile.read(length).decode('utf-8'))
                    if path == '/api/control/mode':
                        if not isinstance(submitted.get('manual'), bool):
                            raise ValueError('manual 必须为布尔值')
                        result = self.dashboard.set_manual_mode(
                            submitted['manual'])
                    else:
                        result = self.dashboard.start_manual_target(
                            submitted.get('distance_m'),
                            submitted.get('angle_deg'))
                self._send_json(200, {'ok': True, **result})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {'ok': False, 'error': str(exc)})
            except Exception as exc:
                logger.exception('手动里程控制失败')
                self._send_json(500, {'ok': False, 'error': str(exc)})
            return
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

    def _stream_mjpeg(self, traffic=False):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        sequence = -1
        try:
            while True:
                previous = sequence
                wait = self.dashboard.wait_for_traffic_frame if traffic else self.dashboard.wait_for_frame
                sequence, jpeg, running = wait(sequence)
                if not running:
                    break
                if jpeg is None or sequence == previous:
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
