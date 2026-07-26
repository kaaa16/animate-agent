const EARTH_ROTATION_SPEED = 0.0016;
const EARTH_AXIS_TILT = -0.36;
const EARTH_DOT_COUNT = 260;
const EARTH_ATMOSPHERE_ALPHA = 0.28;
const EARTH_GRID_ALPHA = 0.16;
const STAR_COUNT = 190;
const STAR_DRIFT_SPEED = 0.012;
const MAX_TEXT_PREVIEW_CHARS = 9000;

const samples = {
  robot:
    "移动机器人避障系统由底盘、电机控制器、2D 激光雷达和避障控制程序组成。雷达持续扫描前方空间，控制器把距离数组转成安全区域，并在障碍物进入阈值时降低速度、比较左右空间、选择更空旷的一侧绕行。",
  transformer:
    "Transformer 的 self-attention 会把每个 token 同时映射成 Query、Key、Value。Query 与所有 Key 计算相似度，softmax 后得到注意力权重，再加权汇总 Value，让模型知道当前词应该关注上下文里的哪些信息。",
  repo:
    "Intuition Engine Agent turns introductions, manuals, papers, and project docs into interactive knowledge movies. It extracts core entities, causal relationships, timeline beats, visual metaphors, and interaction controls, then renders a controlled 2D/3D animation spec instead of arbitrary generated code."
};

const input = document.querySelector("#sourceInput");
const promptBox = document.querySelector("#promptBox");
const fileInput = document.querySelector("#fileInput");
const addFileBtn = document.querySelector("#addFileBtn");
const fileStatus = document.querySelector("#fileStatus");
const resultTitle = document.querySelector("#resultTitle");
const resultSummary = document.querySelector("#resultSummary");
const earthCanvas = document.querySelector("#earthCanvas");
const starCanvas = document.querySelector("#starfield");
const earthCtx = earthCanvas.getContext("2d");
const starCtx = starCanvas.getContext("2d");

let stars = [];
let earthDots = [];
let rotation = 0;
let lastFrame = 0;

function createStars() {
  stars = Array.from({ length: STAR_COUNT }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: 0.45 + Math.random() * 1.5,
    a: 0.28 + Math.random() * 0.72,
    drift: 0.45 + Math.random() * 1.8
  }));
}

function createEarthDots() {
  earthDots = [];
  const clusters = [
    { lat: 44, lon: -98, spread: 32, count: 46 },
    { lat: -12, lon: -58, spread: 27, count: 32 },
    { lat: 50, lon: 16, spread: 28, count: 46 },
    { lat: 24, lon: 78, spread: 34, count: 52 },
    { lat: -24, lon: 134, spread: 22, count: 24 },
    { lat: 4, lon: 22, spread: 26, count: 34 }
  ];

  clusters.forEach((cluster) => {
    for (let i = 0; i < cluster.count; i += 1) {
      const lat = cluster.lat + (Math.random() - 0.5) * cluster.spread;
      const lon = cluster.lon + (Math.random() - 0.5) * cluster.spread * 1.4;
      earthDots.push({ lat: toRad(lat), lon: toRad(lon), size: 1.1 + Math.random() * 2.4 });
    }
  });

  while (earthDots.length < EARTH_DOT_COUNT) {
    earthDots.push({
      lat: toRad(-58 + Math.random() * 116),
      lon: toRad(-180 + Math.random() * 360),
      size: 0.75 + Math.random() * 1.55
    });
  }
}

function resizeCanvas(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resizeAll() {
  resizeCanvas(earthCanvas, earthCtx);
  resizeCanvas(starCanvas, starCtx);
}

function toRad(value) {
  return (value * Math.PI) / 180;
}

function projectPoint(lat, lon, radius, centerX, centerY) {
  const adjustedLon = lon + rotation;
  const x = Math.cos(lat) * Math.sin(adjustedLon);
  const y = Math.sin(lat) * Math.cos(EARTH_AXIS_TILT) - Math.cos(lat) * Math.cos(adjustedLon) * Math.sin(EARTH_AXIS_TILT);
  const z = Math.sin(lat) * Math.sin(EARTH_AXIS_TILT) + Math.cos(lat) * Math.cos(adjustedLon) * Math.cos(EARTH_AXIS_TILT);
  return { x: centerX + x * radius, y: centerY - y * radius, z };
}

function drawStarfield(time) {
  const width = starCanvas.clientWidth;
  const height = starCanvas.clientHeight;
  starCtx.clearRect(0, 0, width, height);

  stars.forEach((star) => {
    const x = ((star.x * width + time * STAR_DRIFT_SPEED * star.drift) % (width + 20)) - 10;
    const y = star.y * height;
    const pulse = 0.58 + Math.sin(time * 0.002 + star.x * 9) * 0.34;
    starCtx.beginPath();
    starCtx.fillStyle = `rgba(230, 241, 255, ${star.a * pulse})`;
    starCtx.arc(x, y, star.r, 0, Math.PI * 2);
    starCtx.fill();
  });
}

function drawEarth() {
  const width = earthCanvas.clientWidth;
  const height = earthCanvas.clientHeight;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.38;

  earthCtx.clearRect(0, 0, width, height);

  const glow = earthCtx.createRadialGradient(centerX, centerY, radius * 0.5, centerX, centerY, radius * 1.45);
  glow.addColorStop(0, "rgba(126, 231, 255, 0.16)");
  glow.addColorStop(0.58, "rgba(70, 111, 225, 0.2)");
  glow.addColorStop(1, "rgba(70, 111, 225, 0)");
  earthCtx.fillStyle = glow;
  earthCtx.beginPath();
  earthCtx.arc(centerX, centerY, radius * 1.45, 0, Math.PI * 2);
  earthCtx.fill();

  const body = earthCtx.createRadialGradient(centerX - radius * 0.34, centerY - radius * 0.28, radius * 0.1, centerX, centerY, radius);
  body.addColorStop(0, "#5fb7ff");
  body.addColorStop(0.42, "#174c9f");
  body.addColorStop(1, "#051a42");
  earthCtx.fillStyle = body;
  earthCtx.beginPath();
  earthCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  earthCtx.fill();

  earthCtx.save();
  earthCtx.beginPath();
  earthCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  earthCtx.clip();
  drawEarthGrid(radius, centerX, centerY);
  drawEarthDots(radius, centerX, centerY);
  earthCtx.restore();

  earthCtx.lineWidth = 2;
  earthCtx.strokeStyle = `rgba(126, 231, 255, ${EARTH_ATMOSPHERE_ALPHA})`;
  earthCtx.beginPath();
  earthCtx.arc(centerX, centerY, radius + 1, 0, Math.PI * 2);
  earthCtx.stroke();
}

function drawEarthGrid(radius, centerX, centerY) {
  earthCtx.strokeStyle = `rgba(224, 246, 255, ${EARTH_GRID_ALPHA})`;
  earthCtx.lineWidth = 1;

  for (let lat = -60; lat <= 60; lat += 30) {
    earthCtx.beginPath();
    let started = false;
    for (let lon = -180; lon <= 180; lon += 4) {
      const p = projectPoint(toRad(lat), toRad(lon), radius, centerX, centerY);
      if (p.z < -0.08) {
        started = false;
        continue;
      }
      if (!started) {
        earthCtx.moveTo(p.x, p.y);
        started = true;
      } else {
        earthCtx.lineTo(p.x, p.y);
      }
    }
    earthCtx.stroke();
  }

  for (let lon = -150; lon <= 180; lon += 30) {
    earthCtx.beginPath();
    let started = false;
    for (let lat = -80; lat <= 80; lat += 4) {
      const p = projectPoint(toRad(lat), toRad(lon), radius, centerX, centerY);
      if (p.z < -0.08) {
        started = false;
        continue;
      }
      if (!started) {
        earthCtx.moveTo(p.x, p.y);
        started = true;
      } else {
        earthCtx.lineTo(p.x, p.y);
      }
    }
    earthCtx.stroke();
  }
}

function drawEarthDots(radius, centerX, centerY) {
  earthDots
    .map((dot) => ({ ...dot, ...projectPoint(dot.lat, dot.lon, radius, centerX, centerY) }))
    .filter((dot) => dot.z > -0.02)
    .sort((a, b) => a.z - b.z)
    .forEach((dot) => {
      const alpha = 0.18 + Math.max(0, dot.z) * 0.58;
      earthCtx.fillStyle = `rgba(143, 245, 196, ${alpha})`;
      earthCtx.beginPath();
      earthCtx.arc(dot.x, dot.y, dot.size * (0.62 + dot.z * 0.38), 0, Math.PI * 2);
      earthCtx.fill();
    });
}

function analyzeText(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    resultTitle.textContent = "等待输入";
    resultSummary.textContent = "输入内容后，这里会预览 Agent 将如何拆解教学动画。";
    return;
  }

  const lower = trimmed.toLowerCase();
  const isCode = /github|repository|api|sdk|install|cli|function|class|package/.test(lower);
  const isModel = /attention|transformer|token|embedding|neural|model|query|key|value/i.test(trimmed);
  const isRobot = /robot|雷达|避障|ros|传感器|底盘|路径/.test(lower);
  const title = isRobot ? "空间决策动画" : isModel ? "模型机制动画" : isCode ? "项目架构动画" : "概念教学动画";
  const nouns = extractKeywords(trimmed).slice(0, 5).join("、") || "对象、关系、状态";

  resultTitle.textContent = title;
  resultSummary.textContent = `将抽取 ${nouns}，生成“对象入场 → 关系建立 → 状态变化 → 交互参数 → 总结镜头”的 2D/3D 分镜。`;
}

function extractKeywords(text) {
  return text
    .replace(/[^\p{Script=Han}A-Za-z0-9_\- ]/gu, " ")
    .split(/\s+/)
    .filter((word) => word.length >= 2)
    .filter((word, index, array) => array.indexOf(word) === index);
}

function readFile(file) {
  fileStatus.textContent = `正在导入：${file.name}`;
  const extension = file.name.split(".").pop().toLowerCase();
  const isLikelyText = /^(txt|md|markdown|json|csv|py|js|ts|tsx|html|css)$/i.test(extension) || file.type.startsWith("text/");

  if (!isLikelyText) {
    input.value = `已导入文件：${file.name}\n类型：${file.type || extension.toUpperCase()}\n\n这个前端原型已接收文件。后续接入后端解析器后，可以在这里抽取 PDF、PPT、Word 或项目 introduction 的正文内容。`;
    fileStatus.textContent = `已选择：${file.name}`;
    analyzeText(input.value);
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result || "").slice(0, MAX_TEXT_PREVIEW_CHARS);
    input.value = text;
    fileStatus.textContent = `已导入：${file.name}`;
    analyzeText(text);
  };
  reader.onerror = () => {
    fileStatus.textContent = `导入失败：${file.name}`;
  };
  reader.readAsText(file);
}

function bindEvents() {
  addFileBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) readFile(file);
  });

  promptBox.addEventListener("submit", (event) => {
    event.preventDefault();
    analyzeText(input.value);
  });

  input.addEventListener("input", () => analyzeText(input.value));

  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = samples[button.dataset.sample];
      fileStatus.textContent = "已载入示例文本";
      analyzeText(input.value);
    });
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    promptBox.addEventListener(eventName, (event) => {
      event.preventDefault();
      promptBox.classList.add("dragging");
      fileStatus.textContent = "松开即可导入文件";
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    promptBox.addEventListener(eventName, (event) => {
      event.preventDefault();
      promptBox.classList.remove("dragging");
    });
  });

  promptBox.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) readFile(file);
  });

  window.addEventListener("resize", resizeAll);
}

function loop(time) {
  const delta = lastFrame ? time - lastFrame : 16;
  lastFrame = time;
  rotation += delta * EARTH_ROTATION_SPEED;
  drawStarfield(time);
  drawEarth();
  requestAnimationFrame(loop);
}

function init() {
  createStars();
  createEarthDots();
  bindEvents();
  resizeAll();
  input.value = samples.robot;
  analyzeText(input.value);
  requestAnimationFrame(loop);
}

init();
