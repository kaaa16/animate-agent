# 技术架构

## 数据流（模板驱动方案 — 默认）

```
用户上传文件 (PPTX/DOCX/PDF)
    │
    ▼
parser.py: extract_text()
    │  返回纯文本字符串
    ▼
script_generator.py: generate_script()
    │
    ├─ [Step 1] Kimi (kimi-k2.7-code) — 场景拆分 ~30s
    │     输入: 文档文本
    │     输出: JSON [{scene, title, narration, key_points}]
    │
    └─ [Step 2] DeepSeek (deepseek-v4-pro) — 场景优化 ~5s
          输入: 文本 + Kimi 场景
          输出: 优化 JSON
    │
    ▼
script_generator.py: generate_lesson_json()    ← 【核心新方案】
    │
    └─ [Step 3] DeepSeek (deepseek-v4-pro) — ~40s
          System: "3D 教学动画设计师" — 选择场景类型，输出 lesson JSON
          输入: 文档文本 + 场景 JSON
          输出: {meta, scene: {type, objects...}, steps: [{objectStates...}]}
          temperature=0.5, max_tokens=4096 (非流式)
    │
    ▼
server.py: POST /api/generate?mode=template
    返回 JSON: { scenes, lesson_json, mode: "template" }
    │
    ▼
frontend/index.html → iframe src="/player-template.html"
    │  postMessage({ type: "lesson-data", payload: lesson_json })
    ▼
player-template.html (Three.js 模板渲染)
    │  场景构建器注册表:
    │    'lidar_obstacle_avoidance' → 地面+小车+LiDAR锥+障碍物+路径标记
    │    'generic_3d' → 地面+灯光（兜底）
    │  步骤切换 → 修改 objectStates → 动画循环更新
    ▼
  window.changeScene(n) ← 父页面导航调用
```

## 兜底方案（mode=2d/3d）

```
generate_script() → generate_animation_html()
    → AI 直接生成完整 HTML (stream=True, ~190s)
    → 前端 iframe srcdoc 注入
```

此方案仍可用，在模板渲染失败或需要极高自由度时作为兜底。

## API 端点

### `GET /` — 前端页面

### `GET /player-template.html` — 3D 模板播放器

### `GET /api/health` — 健康检查

### `POST /api/generate` — 核心接口
- **参数**: 
  - `file` (form-data): 上传文件
  - `mode` (query string): `"template"` (默认) | `"2d"` | `"3d"`
- **template 模式返回**: `{ scenes, lesson_json, mode }`
- **2d/3d 模式返回**: `{ scenes, animation_html, mode }`
- **错误**: 400 (格式/大小), 504 (超时), 500 (其他)

## 前端状态机

```
STATE.UPLOAD  ←──────────────┐
     │ 上传文件               │ 点"取消"/AbortError
     ▼                        │
STATE.LOADING ──── 超时 ──────┤ (显示错误卡片)
     │ 轮询 + 进度条动画       │ 点"重新上传"
     ▼                        │
STATE.PLAYER ─────────────────┘
     │ iframe 渲染动画 HTML
     │ 底部导航按钮切换场景 (调用 iframe.contentWindow.changeScene(n))
     │ 点"重新上传" → UPLOAD
```

关键 JS 变量:
- `TIMEOUT_MS = 600000` (10分钟)
- `currentMode = '3d'` (默认)
- `abortController` + `abortedByTimeout` (取消/超时控制)
- `switchState(state)` (状态切换 + loading 动画启停)

## AI 模型配置

| 用途 | 模型 | API Base | 超时 | 流式 |
|------|------|----------|------|------|
| 场景拆分 | Kimi kimi-k2.7-code | api.moonshot.cn/v1 | 120s | 否 |
| 场景优化 | DeepSeek deepseek-v4-pro | api.deepseek.com | 120s | 否 |
| lesson JSON 生成 | DeepSeek deepseek-v4-pro | api.deepseek.com | 120s | 否 |
| 动画 HTML 生成(兜底) | DeepSeek deepseek-v4-pro | api.deepseek.com | 180s | 是 |

## 三种模式对比

| | Template (默认) | 3D | 2D |
|---|---|---|---|
| 技术 | Three.js 固定模板 | AI 生成 Three.js HTML | AI 生成 Canvas 2D HTML |
| 生成耗时 | ~40s | ~190s | ~190s |
| 渲染质量 | 100% 稳定 | AI 波动 | AI 波动 |
| 灵活性 | 场景类型决定 | 极高（AI 从零写） | 极高（AI 从零写） |
| 新增场景 | 加模板 + prompt | 无需改动 | 无需改动 |
| 推荐场景 | 教学主要内容 | 特殊视角/效果 | 2D 风格偏好 |

## 关键错误处理
- `APITimeoutError` → 504 + 中文提示
- `RateLimitError` (429) → 自动等 3s 重试
- 前端 `AbortError` → 区分超时(保持错误卡片) / 用户取消(回上传页)
- 文件格式/大小校验 → 400 + 中文提示
