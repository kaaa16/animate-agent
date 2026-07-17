const manualSamples = {
  robot: `移动机器人避障手册节选

系统由底盘、电机控制器、2D 激光雷达和避障控制程序组成。激光雷达每 80ms 扫描一次前方 180 度区域，返回每个方向上的距离。控制器需要持续读取雷达数据，并找出前方最近的障碍物。

当最近障碍物距离大于安全距离时，小车保持当前速度直行。当障碍物距离小于安全距离时，小车进入避障状态：先降低速度，再比较左侧和右侧的可通行空间，选择更空旷的一侧转向。完成绕行后，如果正前方重新变得安全，小车恢复直行。

关键参数：
- 安全距离：0.8 米
- 建议速度：0.4 到 1.2 米/秒
- 雷达有效半径：2 到 8 米
- 控制循环：80ms

异常处理：如果雷达连续三次无数据，小车必须停车，并提示传感器异常。`,
  ros: `ROS Publisher 与 Subscriber 手册节选

ROS 中的节点通过 Topic 交换消息。发布者节点负责向指定 Topic 发布消息，订阅者节点监听同一个 Topic 并在收到消息时执行回调函数。

示例：
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
sub = rospy.Subscriber("/scan", LaserScan, scan_callback)

当控制节点发布 /cmd_vel 消息时，底盘节点接收速度指令并驱动机器人移动。当激光雷达节点发布 /scan 消息时，避障节点读取距离数组并判断是否需要减速或转向。`,
  api: `开放平台 API 鉴权手册节选

客户端调用接口前，需要先使用 app_id 与 secret 换取 access_token。服务端校验签名、时间戳和权限范围。如果 token 有效，请求进入业务服务；如果 token 过期，客户端需要刷新 token 后重试。

核心流程：
1. 客户端生成时间戳和签名。
2. 鉴权网关校验签名是否匹配。
3. 校验 token 是否过期。
4. 校验 scope 是否覆盖目标接口。
5. 通过后转发到业务服务。

常见错误：401 表示 token 无效，403 表示权限不足，429 表示请求过快。`
};

const storyboards = {
  robot: [
    ["扫描", "雷达扇形扫过前方，射线把空间切成多个方向。"],
    ["测距", "每条射线返回距离，最近的回波会被高亮。"],
    ["判断", "最近距离低于安全距离，控制器进入避障状态。"],
    ["转向", "比较左右空旷程度，选择更安全的一侧绕行。"],
    ["恢复", "正前方重新安全，小车回到目标路径。"]
  ],
  api: [
    ["签名", "客户端把 app_id、时间戳、secret 组合成签名。"],
    ["校验", "网关检查签名是否能和请求内容对上。"],
    ["过期", "token 超时会被拦截，并触发刷新流程。"],
    ["权限", "scope 不覆盖接口时返回 403。"],
    ["转发", "鉴权通过后，请求进入业务服务。"]
  ],
  ros: [
    ["图谱", "AI 从手册中抽取 Publisher、Topic、Subscriber 和消息类型。"],
    ["发布", "Publisher 节点把 Twist 消息写入 /cmd_vel。"],
    ["飞行", "消息包沿 Topic 通道移动，颜色表示状态变化。"],
    ["源码", "点击 Topic 弹出源码，点击源码反向高亮相关节点。"],
    ["模拟", "拖动机器人，运行状态和指令反馈实时变化。"]
  ]
};

const explainSteps = [
  "第一步：雷达不是看见整张地图，而是沿多个角度测距。画面中的青色射线就是这些测量方向。",
  "第二步：最近的障碍物会变成红色命中点。这个点决定当前是否危险。",
  "第三步：黄色安全圈代表控制器的阈值。障碍进入圈内时，小车不能继续直行。",
  "第四步：控制器比较左右两边的空旷程度，然后输出转向角，车身会朝更空的一侧偏转。",
  "第五步：绕过障碍后，车头逐渐回正，轨迹线显示整个决策造成的运动结果。"
];

const state = {
  mode: "robot",
  playing: true,
  step: 0,
  t: 0,
  car: { x: 120, y: 310, angle: 0, trail: [] },
  ros: {
    shot: 0,
    sourceVisible: false,
    sourceSelected: false,
    highlightedNode: null,
    ncuPulse: 0,
    robot: { x: 500, y: 455 },
    messageProgress: 0,
    hitRegions: {}
  },
  obstacles: [],
  lastHit: null,
  avoidSide: 1,
  dragging: null,
  phaseIndex: -1
};

const els = {
  input: document.querySelector("#manualInput"),
  loadSample: document.querySelector("#loadSampleBtn"),
  analyze: document.querySelector("#analyzeBtn"),
  rosSample: document.querySelector("#rosSampleBtn"),
  apiSample: document.querySelector("#apiSampleBtn"),
  docType: document.querySelector("#docType"),
  objects: document.querySelector("#objectsText"),
  logic: document.querySelector("#logicText"),
  params: document.querySelector("#paramsText"),
  storyboard: document.querySelector("#storyboard"),
  canvas: document.querySelector("#demoCanvas"),
  phase: document.querySelector("#phaseText"),
  decision: document.querySelector("#decisionText"),
  play: document.querySelector("#playBtn"),
  reset: document.querySelector("#resetBtn"),
  speed: document.querySelector("#speedRange"),
  lidar: document.querySelector("#lidarRange"),
  safe: document.querySelector("#safeRange"),
  density: document.querySelector("#densityRange"),
  step: document.querySelector("#stepBtn"),
  explain: document.querySelector("#explainText"),
  title: document.querySelector("#demoTitle")
};

const ctx = els.canvas.getContext("2d");

function init() {
  els.input.value = manualSamples.robot;
  resetObstacles();
  renderStoryboard();
  bindEvents();
  resizeCanvas();
  requestAnimationFrame(loop);
}

function bindEvents() {
  els.loadSample.addEventListener("click", () => {
    state.mode = "robot";
    els.input.value = manualSamples.robot;
    analyzeManual();
    resetDemo();
  });

  els.apiSample.addEventListener("click", () => {
    state.mode = "api";
    els.input.value = manualSamples.api;
    analyzeManual();
  });

  els.rosSample.addEventListener("click", () => {
    state.mode = "ros";
    els.input.value = manualSamples.ros;
    analyzeManual();
    resetRosDemo();
  });

  els.analyze.addEventListener("click", analyzeManual);
  els.play.addEventListener("click", () => {
    state.playing = !state.playing;
    els.play.textContent = state.playing ? "Ⅱ" : "▶";
  });
  els.reset.addEventListener("click", resetDemo);
  els.step.addEventListener("click", () => {
    state.step = (state.step + 1) % explainSteps.length;
    els.explain.textContent = explainSteps[state.step];
  });
  els.density.addEventListener("input", resetObstacles);

  els.canvas.addEventListener("pointerdown", onPointerDown);
  els.canvas.addEventListener("pointermove", onPointerMove);
  els.canvas.addEventListener("click", onCanvasClick);
  els.canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
  window.addEventListener("pointerup", () => {
    state.dragging = null;
  });
  window.addEventListener("resize", resizeCanvas);
}

function analyzeManual() {
  const text = els.input.value;
  const isRos = /ROS|rospy|Publisher|Subscriber|Topic|cmd_vel|LaserScan|Twist/i.test(text);
  const isApi = /token|签名|鉴权|scope|接口|401|403/i.test(text);
  state.mode = isRos ? "ros" : isApi ? "api" : "robot";
  els.docType.textContent = isRos ? "ROS 通信手册" : isApi ? "API 鉴权手册" : "机器人手册";
  els.title.textContent = isRos
    ? "ROS 交互式知识电影"
    : isApi
      ? "API 鉴权流程动画实验台"
      : "小车避障动画实验台";

  if (isRos) {
    els.objects.textContent = "Publisher / Topic / Subscriber / Message / Robot Runtime";
    els.logic.textContent = "节点不直接互相调用，而是通过 Topic 传递带类型的消息";
    els.params.textContent = "Topic 名称、消息类型、队列、回调函数、机器人运行状态";
    els.explain.textContent = "滚轮推进镜头；点击 /cmd_vel 查看源码；点击源码反向高亮节点；拖动机器人观察运行状态。";
  } else if (isApi) {
    els.objects.textContent = "客户端 / 鉴权网关 / Token / Scope / 业务服务";
    els.logic.textContent = "请求必须依次通过签名、过期时间、权限范围校验";
    els.params.textContent = "token 有效期、scope、重试次数、限流阈值";
    els.explain.textContent = "API 文档适合转成链路动画：请求包穿过网关，每个校验节点决定放行、刷新或拒绝。";
  } else {
    els.objects.textContent = "小车 / 激光雷达 / 障碍物 / 安全距离 / 控制器";
    els.logic.textContent = "最近距离低于安全阈值时，控制器降速并选择空旷方向";
    els.params.textContent = "车速、雷达半径、安全距离、障碍密度";
    els.explain.textContent = explainSteps[0];
  }
  setControlsForMode();
  renderStoryboard();
}

function renderStoryboard() {
  els.storyboard.innerHTML = "";
  const phase = currentPhaseIndex();
  storyboards[state.mode].forEach(([title, body], index) => {
    const card = document.createElement("article");
    card.className = `story-card ${index === phase ? "active" : ""}`;
    card.innerHTML = `<b>${String(index + 1).padStart(2, "0")} / ${title}</b><p>${body}</p>`;
    els.storyboard.appendChild(card);
  });
}

function resetDemo() {
  state.t = 0;
  state.step = 0;
  state.car = { x: 120, y: 310, angle: 0, trail: [] };
  state.playing = true;
  state.phaseIndex = -1;
  els.play.textContent = "Ⅱ";
  resetObstacles();
  resetRosDemo();
  renderStoryboard();
}

function resetRosDemo() {
  state.ros = {
    shot: 0,
    sourceVisible: false,
    sourceSelected: false,
    highlightedNode: null,
    ncuPulse: 0,
    robot: { x: 500, y: 455 },
    messageProgress: 0,
    hitRegions: {}
  };
  state.phaseIndex = -1;
}

function setControlsForMode() {
  const deck = document.querySelector(".control-deck");
  if (!deck) return;
  deck.classList.toggle("disabled", state.mode === "ros" || state.mode === "api");
}

function setPhase(phase) {
  if (phase === state.phaseIndex) return;
  state.phaseIndex = phase;
  renderStoryboard();
}

function resetObstacles() {
  const count = Number(els.density.value);
  const base = [
    { x: 410, y: 300, r: 34 },
    { x: 570, y: 210, r: 27 },
    { x: 620, y: 410, r: 42 },
    { x: 780, y: 310, r: 32 },
    { x: 730, y: 150, r: 25 },
    { x: 880, y: 470, r: 31 },
    { x: 500, y: 485, r: 24 }
  ];
  state.obstacles = base.slice(0, count).map((item) => ({ ...item }));
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(640, Math.floor(rect.width * scale));
  els.canvas.height = Math.max(360, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function loop() {
  if (state.playing && state.mode === "robot") {
    updateRobot();
  }
  if (state.playing && state.mode === "ros") {
    updateRos();
  }
  draw();
  requestAnimationFrame(loop);
}

function updateRos() {
  state.t += 0.018;
  state.ros.messageProgress = (state.ros.messageProgress + 0.006) % 1;
  if (state.ros.ncuPulse > 0) state.ros.ncuPulse = Math.max(0, state.ros.ncuPulse - 0.025);
  els.phase.textContent = `ROS 镜头 ${state.ros.shot + 1} / ${storyboards.ros[state.ros.shot][0]}`;
  const speedHint = Math.round((state.ros.robot.x - 500) / 8) / 10;
  els.decision.textContent = `机器人实时模拟：linear.x=${speedHint.toFixed(1)}，Topic=/cmd_vel`;
  setPhase(state.ros.shot);
}

function updateRobot() {
  const speed = Number(els.speed.value);
  const safe = Number(els.safe.value);
  state.t += 0.016 * speed;

  const hit = nearestObstacle();
  state.lastHit = hit;
  const danger = hit && hit.distance < safe + 38;
  const openTop = clearanceAt(-1);
  const openBottom = clearanceAt(1);
  state.avoidSide = openBottom > openTop ? 1 : -1;

  const targetAngle = danger ? state.avoidSide * 0.62 : (310 - state.car.y) * 0.004;
  state.car.angle += (targetAngle - state.car.angle) * 0.055;
  state.car.x += Math.cos(state.car.angle) * (1.35 + speed);
  state.car.y += Math.sin(state.car.angle) * (1.35 + speed);

  if (state.car.x > 940) {
    state.car.x = 105;
    state.car.y = 310;
    state.car.angle = 0;
    state.car.trail = [];
  }
  state.car.y = Math.max(95, Math.min(520, state.car.y));
  state.car.trail.push({ x: state.car.x, y: state.car.y });
  if (state.car.trail.length > 210) state.car.trail.shift();

  updateHud(danger, hit);
}

function nearestObstacle() {
  const radius = Number(els.lidar.value);
  let best = null;
  for (const obstacle of state.obstacles) {
    const dx = obstacle.x - state.car.x;
    const dy = obstacle.y - state.car.y;
    const forward = dx * Math.cos(state.car.angle) + dy * Math.sin(state.car.angle);
    const side = Math.abs(-dx * Math.sin(state.car.angle) + dy * Math.cos(state.car.angle));
    const dist = Math.hypot(dx, dy) - obstacle.r;
    if (forward > 0 && forward < radius && side < 120 && (!best || dist < best.distance)) {
      best = { obstacle, distance: dist, x: obstacle.x, y: obstacle.y };
    }
  }
  return best;
}

function clearanceAt(side) {
  let score = 220;
  for (const obstacle of state.obstacles) {
    const dy = side * (obstacle.y - state.car.y);
    const dx = obstacle.x - state.car.x;
    if (dx > 0 && dx < 260 && dy > 0) {
      score -= Math.max(0, 120 - Math.abs(dy));
    }
  }
  return score;
}

function updateHud(danger, hit) {
  const phase = currentPhaseIndex();
  const labels = ["阶段 1 / 雷达扫描", "阶段 2 / 距离回波", "阶段 3 / 阈值判断", "阶段 4 / 转向绕行", "阶段 5 / 恢复直行"];
  els.phase.textContent = labels[phase];
  if (!hit) {
    els.decision.textContent = "前方未发现障碍，保持目标路径";
  } else if (danger) {
    els.decision.textContent = `最近障碍 ${Math.round(hit.distance)}px，低于安全阈值，向${state.avoidSide > 0 ? "下" : "上"}绕行`;
  } else {
    els.decision.textContent = `最近障碍 ${Math.round(hit.distance)}px，仍在安全范围外`;
  }
  setPhase(phase);
}

function currentPhaseIndex() {
  if (state.mode === "api") return Math.floor((Date.now() / 1200) % 5);
  if (!state.lastHit) return 0;
  const safe = Number(els.safe.value);
  if (state.lastHit.distance > safe + 70) return 1;
  if (state.lastHit.distance > safe + 20) return 2;
  if (Math.abs(state.car.angle) > 0.23) return 3;
  return 4;
}

function draw() {
  const rect = els.canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (state.mode === "api") {
    drawApiFlow(rect.width, rect.height);
  } else if (state.mode === "ros") {
    drawRosMovie(rect.width, rect.height);
  } else {
    drawRobot(rect.width, rect.height);
  }
}

function drawRosMovie(width, height) {
  drawGrid(width, height);
  const nodes = {
    publisher: { id: "publisher", label: "Publisher\n/talker", x: 150, y: 210, w: 145, h: 86 },
    topic: { id: "topic", label: "Topic\n/cmd_vel", x: width / 2 - 82, y: 205, w: 164, h: 96 },
    subscriber: { id: "subscriber", label: "Subscriber\n/base_controller", x: width - 270, y: 210, w: 190, h: 86 }
  };
  state.ros.hitRegions = {};
  drawKnowledgeGraph(nodes);
  drawRosMessage(nodes);
  drawSourcePanel(width, height);
  drawRuntimeRobot(height);
  drawMovieCaption(width);
}

function drawKnowledgeGraph(nodes) {
  ctx.save();
  drawRosEdge(nodes.publisher, nodes.topic);
  drawRosEdge(nodes.topic, nodes.subscriber);
  drawRosNode(nodes.publisher, "publisher");
  drawRosTopic(nodes.topic);
  drawRosNode(nodes.subscriber, "subscriber");
  ctx.restore();
}

function drawRosEdge(from, to) {
  const active = state.ros.shot >= 1;
  ctx.save();
  ctx.strokeStyle = active ? "rgba(66, 247, 255, 0.7)" : "rgba(66, 247, 255, 0.18)";
  ctx.lineWidth = active ? 3 : 1.5;
  ctx.setLineDash(active ? [] : [8, 12]);
  ctx.beginPath();
  ctx.moveTo(from.x + from.w, from.y + from.h / 2);
  ctx.lineTo(to.x, to.y + to.h / 2);
  ctx.stroke();
  ctx.restore();
}

function drawRosNode(node, key) {
  const highlighted = state.ros.highlightedNode === key || state.ros.sourceSelected;
  const pulse = highlighted ? 1 + state.ros.ncuPulse * 0.18 : 1;
  const x = node.x + (node.w - node.w * pulse) / 2;
  const y = node.y + (node.h - node.h * pulse) / 2;
  const w = node.w * pulse;
  const h = node.h * pulse;
  ctx.save();
  ctx.fillStyle = highlighted ? "rgba(255, 47, 214, 0.36)" : "rgba(5, 9, 16, 0.88)";
  ctx.strokeStyle = highlighted ? "#ff2fd6" : "#42f7ff";
  ctx.lineWidth = highlighted ? 3 : 2;
  roundRect(x, y, w, h, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ecfbff";
  ctx.font = "800 15px Microsoft YaHei, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  node.label.split("\n").forEach((line, index) => {
    ctx.fillText(line, node.x + node.w / 2, node.y + 32 + index * 24);
  });
  ctx.fillStyle = key === "publisher" ? "#5dff9c" : "#f4f06d";
  ctx.fillRect(node.x + 12, node.y + node.h - 14, node.w - 24, 4);
  ctx.restore();
  state.ros.hitRegions[key] = node;
}

function drawRosTopic(topic) {
  const active = state.ros.sourceVisible || state.ros.shot >= 2;
  ctx.save();
  ctx.fillStyle = active ? "rgba(244, 240, 109, 0.16)" : "rgba(5, 9, 16, 0.88)";
  ctx.strokeStyle = active ? "#f4f06d" : "#42f7ff";
  ctx.lineWidth = active ? 3 : 2;
  roundRect(topic.x, topic.y, topic.w, topic.h, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4f06d";
  ctx.font = "800 16px Microsoft YaHei, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  topic.label.split("\n").forEach((line, index) => {
    ctx.fillText(line, topic.x + topic.w / 2, topic.y + 34 + index * 26);
  });
  drawCallout(topic.x + 14, topic.y + topic.h + 12, "点击 Topic 查看源码", "#f4f06d");
  ctx.restore();
  state.ros.hitRegions.topic = topic;
}

function drawRosMessage(nodes) {
  const start = { x: nodes.publisher.x + nodes.publisher.w, y: nodes.publisher.y + nodes.publisher.h / 2 };
  const mid = { x: nodes.topic.x + nodes.topic.w / 2, y: nodes.topic.y + nodes.topic.h / 2 };
  const end = { x: nodes.subscriber.x, y: nodes.subscriber.y + nodes.subscriber.h / 2 };
  const p = state.ros.messageProgress;
  const firstLeg = p < 0.5;
  const local = firstLeg ? p * 2 : (p - 0.5) * 2;
  const from = firstLeg ? start : mid;
  const to = firstLeg ? mid : end;
  const x = from.x + (to.x - from.x) * local;
  const y = from.y + (to.y - from.y) * local;
  const received = p > 0.78;
  ctx.save();
  ctx.fillStyle = received ? "#5dff9c" : "#ff2fd6";
  ctx.shadowColor = received ? "#5dff9c" : "#ff2fd6";
  ctx.shadowBlur = 20;
  roundRect(x - 34, y - 17, 68, 34, 7);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#07080d";
  ctx.font = "800 12px Microsoft YaHei, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Twist", x, y);
  ctx.restore();
}

function drawSourcePanel(width, height) {
  if (!state.ros.sourceVisible && state.ros.shot < 3) return;
  const panel = { x: Math.max(24, width - 390), y: 330, w: 354, h: 132 };
  ctx.save();
  ctx.fillStyle = state.ros.sourceSelected ? "rgba(255, 47, 214, 0.18)" : "rgba(5, 9, 16, 0.9)";
  ctx.strokeStyle = state.ros.sourceSelected ? "#ff2fd6" : "#42f7ff";
  ctx.lineWidth = 2;
  roundRect(panel.x, panel.y, panel.w, panel.h, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#42f7ff";
  ctx.font = "800 13px Consolas, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  const lines = [
    'pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)',
    "msg = Twist()",
    "msg.linear.x = 0.4",
    "pub.publish(msg)"
  ];
  lines.forEach((line, index) => {
    ctx.fillStyle = index === 0 ? "#f4f06d" : "#ecfbff";
    ctx.fillText(line, panel.x + 14, panel.y + 16 + index * 25);
  });
  ctx.restore();
  state.ros.hitRegions.source = panel;
}

function drawRuntimeRobot(height) {
  const robot = state.ros.robot;
  const y = Math.min(height - 70, robot.y);
  ctx.save();
  ctx.strokeStyle = "rgba(93, 255, 156, 0.55)";
  ctx.setLineDash([9, 11]);
  ctx.beginPath();
  ctx.moveTo(70, y);
  ctx.lineTo(900, y);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.translate(robot.x, y);
  ctx.fillStyle = "rgba(12, 18, 26, 0.96)";
  ctx.strokeStyle = "#5dff9c";
  ctx.lineWidth = 3;
  roundRect(-44, -24, 88, 48, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4f06d";
  ctx.beginPath();
  ctx.moveTo(50, 0);
  ctx.lineTo(30, -13);
  ctx.lineTo(30, 13);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#42f7ff";
  ctx.beginPath();
  ctx.arc(-10, 0, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  drawCallout(robot.x - 95, y + 36, "拖动机器人实时模拟", "#5dff9c");
  state.ros.hitRegions.robot = { x: robot.x - 50, y: y - 30, w: 100, h: 60 };
}

function drawMovieCaption(width) {
  const [title, body] = storyboards.ros[state.ros.shot];
  ctx.save();
  ctx.fillStyle = "rgba(5, 9, 16, 0.82)";
  ctx.strokeStyle = "rgba(66, 247, 255, 0.42)";
  roundRect(24, 24, Math.min(520, width - 48), 76, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4f06d";
  ctx.font = "900 16px Microsoft YaHei, sans-serif";
  ctx.textBaseline = "top";
  ctx.fillText(`${state.ros.shot + 1}. ${title}`, 42, 38);
  ctx.fillStyle = "#ecfbff";
  ctx.font = "700 13px Microsoft YaHei, sans-serif";
  ctx.fillText(body, 42, 66);
  ctx.restore();
}

function drawRobot(width, height) {
  drawGrid(width, height);
  drawPath();
  drawSafeZone();
  drawLidar();
  drawObstacles();
  drawCar();
  drawDecisionOverlay(width);
}

function drawGrid(width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(66, 247, 255, 0.07)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 34) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 34) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(244, 240, 109, 0.24)";
  ctx.setLineDash([10, 14]);
  ctx.beginPath();
  ctx.moveTo(70, 310);
  ctx.lineTo(width - 60, 310);
  ctx.stroke();
  ctx.restore();
}

function drawPath() {
  if (state.car.trail.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "rgba(93, 255, 156, 0.72)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  state.car.trail.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawSafeZone() {
  ctx.save();
  ctx.translate(state.car.x, state.car.y);
  ctx.strokeStyle = "rgba(244, 240, 109, 0.52)";
  ctx.fillStyle = "rgba(244, 240, 109, 0.045)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 9]);
  ctx.beginPath();
  ctx.arc(0, 0, Number(els.safe.value), -Math.PI * 0.72, Math.PI * 0.72);
  ctx.stroke();
  ctx.fill();
  ctx.restore();
}

function drawLidar() {
  const radius = Number(els.lidar.value);
  ctx.save();
  ctx.translate(state.car.x, state.car.y);
  ctx.rotate(state.car.angle);
  ctx.fillStyle = "rgba(66, 247, 255, 0.055)";
  ctx.strokeStyle = "rgba(66, 247, 255, 0.26)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, radius, -Math.PI / 2.2, Math.PI / 2.2);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  for (let i = -6; i <= 6; i += 1) {
    const angle = (i / 6) * (Math.PI / 2.35);
    const pulse = 0.72 + Math.sin(state.t * 8 + i) * 0.18;
    ctx.strokeStyle = `rgba(66, 247, 255, ${0.18 + pulse * 0.26})`;
    ctx.beginPath();
    ctx.moveTo(16, 0);
    ctx.lineTo(Math.cos(angle) * radius, Math.sin(angle) * radius);
    ctx.stroke();
  }
  ctx.restore();

  if (state.lastHit) {
    ctx.save();
    ctx.strokeStyle = "rgba(255, 94, 122, 0.9)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(state.car.x, state.car.y);
    ctx.lineTo(state.lastHit.x, state.lastHit.y);
    ctx.stroke();
    ctx.fillStyle = "#ff5e7a";
    ctx.beginPath();
    ctx.arc(state.lastHit.x, state.lastHit.y, 6, 0, Math.PI * 2);
    ctx.fill();
    drawCallout(
      state.lastHit.x + 14,
      state.lastHit.y - 18,
      `最近距离 ${Math.max(0, Math.round(state.lastHit.distance))}px`,
      "#ff5e7a"
    );
    ctx.restore();
  }
}

function drawObstacles() {
  for (const obstacle of state.obstacles) {
    const isHit = state.lastHit && state.lastHit.obstacle === obstacle;
    ctx.save();
    ctx.translate(obstacle.x, obstacle.y);
    ctx.fillStyle = isHit ? "rgba(255, 94, 122, 0.82)" : "rgba(255, 47, 214, 0.42)";
    ctx.strokeStyle = isHit ? "#ff5e7a" : "rgba(255, 47, 214, 0.88)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < 8; i += 1) {
      const angle = (Math.PI * 2 * i) / 8 + 0.22;
      const r = obstacle.r * (i % 2 ? 0.78 : 1);
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

function drawCar() {
  ctx.save();
  ctx.translate(state.car.x, state.car.y);
  ctx.rotate(state.car.angle);
  ctx.fillStyle = "rgba(12, 18, 26, 0.96)";
  ctx.strokeStyle = "#42f7ff";
  ctx.lineWidth = 3;
  roundRect(-31, -20, 62, 40, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4f06d";
  ctx.beginPath();
  ctx.moveTo(34, 0);
  ctx.lineTo(17, -11);
  ctx.lineTo(17, 11);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#42f7ff";
  ctx.beginPath();
  ctx.arc(0, 0, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(244, 240, 109, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(48, 0);
  ctx.stroke();
  ctx.fillStyle = "#ff2fd6";
  ctx.fillRect(-22, -24, 16, 5);
  ctx.fillRect(-22, 19, 16, 5);
  ctx.restore();
}

function drawDecisionOverlay(width) {
  const safe = Number(els.safe.value);
  const hit = state.lastHit;
  const danger = hit && hit.distance < safe + 38;
  const maxX = Math.max(18, width - 268);
  const x = Math.min(maxX, Math.max(18, state.car.x + 72));
  const y = Math.max(22, state.car.y - 112);
  const lines = hit
    ? [
        `输入：雷达最近回波 ${Math.round(hit.distance)}px`,
        `阈值：安全距离 ${safe}px`,
        danger ? `输出：减速，向${state.avoidSide > 0 ? "下" : "上"}绕行` : "输出：保持直行"
      ]
    : ["输入：前方无有效回波", `阈值：安全距离 ${safe}px`, "输出：保持直行"];

  ctx.save();
  ctx.fillStyle = "rgba(5, 9, 16, 0.82)";
  ctx.strokeStyle = danger ? "rgba(255, 94, 122, 0.9)" : "rgba(66, 247, 255, 0.55)";
  ctx.lineWidth = 1.5;
  roundRect(x, y, 250, 92, 7);
  ctx.fill();
  ctx.stroke();
  ctx.font = "700 13px Microsoft YaHei, sans-serif";
  ctx.textBaseline = "top";
  lines.forEach((line, index) => {
    ctx.fillStyle = index === 2 ? (danger ? "#ff5e7a" : "#5dff9c") : "#ecfbff";
    ctx.fillText(line, x + 12, y + 12 + index * 24);
  });
  ctx.restore();
}

function drawCallout(x, y, text, color) {
  ctx.save();
  ctx.font = "700 12px Microsoft YaHei, sans-serif";
  const width = ctx.measureText(text).width + 18;
  ctx.fillStyle = "rgba(5, 9, 16, 0.88)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  roundRect(x, y, width, 28, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 9, y + 14);
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawApiFlow(width, height) {
  drawGrid(width, height);
  const nodes = [
    ["Client", 120, height * 0.5],
    ["Signature", 310, height * 0.35],
    ["Token", 500, height * 0.5],
    ["Scope", 690, height * 0.35],
    ["Service", 880, height * 0.5]
  ];
  ctx.save();
  ctx.lineWidth = 3;
  for (let i = 0; i < nodes.length - 1; i += 1) {
    ctx.strokeStyle = "rgba(66, 247, 255, 0.55)";
    ctx.beginPath();
    ctx.moveTo(nodes[i][1] + 54, nodes[i][2]);
    ctx.lineTo(nodes[i + 1][1] - 54, nodes[i + 1][2]);
    ctx.stroke();
  }
  const packet = (Date.now() / 900) % 4;
  const from = nodes[Math.floor(packet)];
  const to = nodes[Math.floor(packet) + 1] || nodes[4];
  const mix = packet - Math.floor(packet);
  const px = from[1] + (to[1] - from[1]) * mix;
  const py = from[2] + (to[2] - from[2]) * mix;
  ctx.fillStyle = "#f4f06d";
  ctx.shadowColor = "#f4f06d";
  ctx.shadowBlur = 18;
  ctx.fillRect(px - 12, py - 8, 24, 16);
  ctx.shadowBlur = 0;

  nodes.forEach(([label, x, y], index) => {
    ctx.fillStyle = index === Math.floor(packet) + 1 ? "rgba(255, 47, 214, 0.42)" : "rgba(6, 12, 20, 0.94)";
    ctx.strokeStyle = index === Math.floor(packet) + 1 ? "#ff2fd6" : "#42f7ff";
    ctx.lineWidth = 2;
    roundRect(x - 56, y - 32, 112, 64, 7);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ecfbff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "700 15px Microsoft YaHei, sans-serif";
    ctx.fillText(label, x, y);
  });
  ctx.restore();
  els.phase.textContent = "API 鉴权链路";
  els.decision.textContent = "请求包逐段通过签名、Token 与 Scope 校验";
  setPhase(Math.floor(packet) + 1 > 4 ? 4 : Math.floor(packet) + 1);
}

function canvasPoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function onPointerDown(event) {
  const point = canvasPoint(event);
  if (state.mode === "ros") {
    const robot = state.ros.hitRegions.robot;
    if (robot && point.x >= robot.x && point.x <= robot.x + robot.w && point.y >= robot.y && point.y <= robot.y + robot.h) {
      state.dragging = "rosRobot";
    }
    return;
  }
  if (state.mode !== "robot") return;
  state.dragging = state.obstacles.find((obstacle) => Math.hypot(obstacle.x - point.x, obstacle.y - point.y) <= obstacle.r + 12);
}

function onPointerMove(event) {
  if (!state.dragging) return;
  const point = canvasPoint(event);
  if (state.dragging === "rosRobot") {
    state.ros.robot.x = Math.max(130, Math.min(850, point.x));
    state.ros.robot.y = Math.max(390, Math.min(540, point.y));
    state.ros.shot = 4;
    return;
  }
  state.dragging.x = Math.max(80, Math.min(920, point.x));
  state.dragging.y = Math.max(80, Math.min(540, point.y));
}

function onCanvasClick(event) {
  if (state.mode !== "ros") return;
  const point = canvasPoint(event);
  const hit = hitRosRegion(point);
  if (hit === "topic") {
    state.ros.sourceVisible = true;
    state.ros.shot = 3;
  } else if (hit === "source") {
    state.ros.sourceSelected = true;
    state.ros.highlightedNode = "publisher";
    state.ros.ncuPulse = 1;
  } else if (hit === "publisher" || hit === "subscriber") {
    state.ros.highlightedNode = hit;
    state.ros.ncuPulse = 1;
  }
}

function onCanvasWheel(event) {
  if (state.mode !== "ros") return;
  event.preventDefault();
  const direction = event.deltaY > 0 ? 1 : -1;
  state.ros.shot = Math.max(0, Math.min(storyboards.ros.length - 1, state.ros.shot + direction));
  if (state.ros.shot >= 3) state.ros.sourceVisible = true;
  setPhase(state.ros.shot);
}

function hitRosRegion(point) {
  for (const [key, region] of Object.entries(state.ros.hitRegions)) {
    if (point.x >= region.x && point.x <= region.x + region.w && point.y >= region.y && point.y <= region.y + region.h) {
      return key;
    }
  }
  return null;
}

init();
