# 开发历史

## 2026-09-10 — IR 中间表示层重构 + DeepSeek 模型分层 + 截断修复

### 引入 IR 中间表示层（向「DocumentIR→LessonIR→StoryboardIR→多渲染器」目标架构靠拢）
- 新建 `backend/ir_schemas.py`，定义三段数据契约：
  - `DocumentIR`（source_type/title/sections/raw_text）—— Ingest 输出
  - `LessonIR`（scenes：scene/title/narration/key_points）—— Knowledge Agent 输出
  - `StoryboardIR`（scenes：scene_type/renderer_hint/objects/steps）—— Storyboard Agent 输出（渲染器无关，暂未接入管线）
- `parser.py`：`read_pptx/read_docx/read_pdf` 改为返回 `DocumentIR`，新增 `extract_document_ir()`；`extract_text()` 仍返回纯文本（兼容）
- `script_generator.py`：`generate_script()` 改为返回 `LessonIR`；新增 `_to_scenes_list()`，`generate_lesson_json` / `generate_lesson_json_2d` / `generate_animation_html` 统一接收 LessonIR/dict/list
- `server.py` + `test_template.py`/`test_template_2d.py`/`test_e2e.py` 同步适配 LessonIR

### DeepSeek 模型分层（pro 保核心、flash 扛次要）
- 新增 `DEEPSEEK_MODEL_FLASH = "deepseek-v4-flash"`（已验证为有效 id）
- 「场景优化」(`_call_deepseek`) 和「兜底动画 HTML」(`generate_animation_html`) 切到 flash
- 核心 lesson JSON 生成保持 `deepseek-v4-pro` 不变

### 修复 lesson JSON 输出截断
- `generate_lesson_json` / `generate_lesson_json_2d` 的 `max_tokens` 4096 → 16384
- 根因：deepseek-v4-pro 在吐 JSON 前消耗大量思考 token，4096 预算被吃光导致 `finish_reason=length`（3 次重试全截断）
- 验证：`test_template.py`（编程主题→generic_3d）此前连续截断失败，改后通过

### 验证
- `test_new_scene.py` / `test_template.py` / `test_template_2d.py` 通过
- 非 API 验证：py_compile、导入、`_to_scenes_list` 三输入、`read_*` 返回 DocumentIR 全部通过

## 2026-09-05 — 斜抛速度分解 + 实时读数 + 暂停键

### 斜抛速度矢量分解（可视化核心增强）
- 3D / 2D 斜抛（`projectile_motion` / `projectile_motion_2d`）新增瞬时速度三矢量：
  - vx（青蓝，水平恒定）、vy（绿，竖直，越过最高点后反向）、v（橙，合速度，沿切线）
  - 三者构成平行四边形（虚线辅助线），直观展示 v = vx + vy 的矢量合成
- 新增「速度读数」固定面板（左上角），实时显示 vx / vy / v 的具体数值（m/s，保留两位）
- 数值随飞行时间变化，也随滑杆调参即时变化（vy 越过最高点由正转负，v 在最高点等于 vx）

### 暂停键
- 右下角新增 ⏸/▶ 按钮：暂停时冻结小球、箭头、读数、背景粒子与最高点脉动
- 暂停后拖滑杆：小球跳到新轨迹对应时刻的位置、读数同步更新（3D 从「跳过 update」改为「不推进时间但仍刷新」，与 2D 行为一致）

### 速度数据准确性
- 采用标准无空气阻力斜抛公式：vx = v0·cosθ（恒定）、vy = v0·sinθ − g·t、v = √(vx² + vy²)
- 已核对默认参数（v0=5, θ=45°, g=5）：发射 vx=3.54 / vy=+3.54 / v=5.00；最高点 vy=0 / v=3.54；落地 vy=−3.54 / v=5.00

### 验证
- 3D + 2D 斜抛：拖滑杆轨迹/最高点/射程/小球/读数即时变化 ✅
- 暂停键冻结/恢复 ✅；暂停时调参，小球位置与读数联动 ✅

## 2026-09-05 — 参数交互（滑杆即时调节）+ 生成健壮性修复

### 参数交互：拖滑杆即时改动画（核心交互能力）
- 3D `frontend/player-template.html`、2D `frontend/player-template-2d.html` 新增「参数调节」面板
- `SCENE_PARAMS` 注册表声明每个场景可调参数（key / label / min / max / step / unit）
- 场景控制器接口扩展：`setParam(key, value)` + `getParams()`
- `projectile_motion` / `projectile_motion_2d` 提供三个滑杆：初速度、发射角度、重力加速度
- `force_analysis` / `force_analysis_2d` 提供三个滑杆：拉力大小、重力大小、摩擦力大小（拖动时箭头长度实时变化）
- 拖动滑杆即时重算物理量（v0x/v0y/飞行时间/射程/最高点）并重建派生物体（轨迹/箭头/最高点/射程标记），小球不从头重放
- `renderParamPanel()` 在 buildScene 末尾按场景类型渲染滑杆

### 后端健壮性：JSON 截断/解析失败自动重试
- 新增 `_parse_lesson_json()`（支持代码块、首尾花括号提取）与 `_call_lesson_json()`（finish_reason=length 或解析失败时最多重试 3 次）
- 解决 DeepSeek 偶发输出截断导致 3D 受力分析崩溃的问题
- `generate_lesson_json` / `generate_lesson_json_2d` 统一走 `_call_lesson_json`

### 验证
- 3D + 2D 抛物运动滑杆手动验证通过：拖动三个滑杆，轨迹/最高点/射程/小球飞行即时变化 ✅
- 3D + 2D 受力分析滑杆手动验证通过：拖动时力箭头长度实时变化 ✅

### 已知待修小问题（低优先级，暂缓）
- 2D 播放器缺拖拽平移（左右拉动画面）
- 3D 播放器缺平移（拖动整个模型移动，当前 OrbitControls `enablePan=false`）
- 参数面板数值框宽度随位数变化抖动（如 8 vs 8.5）

## 2026-09-03 — 物理场景模板 + 场景控制器重构

### 播放器重构为「场景控制器」模式（关键架构升级）
- **3D** `frontend/player-template.html`、**2D** `frontend/player-template-2d.html` 从「硬编码 applyStep/animate」重构为注册表模式
- 每个 `sceneBuilders['type']` 返回自包含控制器：3D 为 `{objects, applyState(step), update(dt,time)}`，2D 为 `{state, applyState(step), draw(time,dt)}`
- 主循环改为通用调用，以后每加一种场景 = 只写一个自包含函数，零改动主循环

### 新增物理场景模板（数学/物理可视化第一批）
- 3D：`projectile_motion`（抛物运动：初速度矢量、虚线抛物线轨迹、最高点/射程标注）、`force_analysis`（受力分析：方块 + 四个力矢量箭头）
- 2D：`projectile_motion_2d`（侧视抛物）、`force_analysis_2d`（正视受力）
- 物理公式与渲染思路借鉴 GitHub（CannonBall 弹道公式、p5.js 力矢量画法），采用**确定性参数方程**而非物理引擎，适配分步教学动画

### 后端 prompt + 白名单扩展
- `LESSON_SYSTEM_PROMPT`（3D）/ `LESSON_SYSTEM_PROMPT_2D` 新增两类场景描述 + 各自 objectStates 参数说明
- 白名单 `valid_3d_types` / `valid_2d_types` 扩展到 4 种类型

### 修复的关键 bug
- `generate_lesson_json_2d` 函数内部存在**第二个独立白名单**（只认 lidar/generic），会把模型正确输出的 `projectile_motion_2d` 静默改写成 `generic_2d` —— 这是 2D 测试长期失败的真正根因（此前误判为 prompt 偏置 / 俯视视角）
- 同步修正 2D prompt：objectStates 按场景分列、去掉全局「俯视视角」限定（lidar=俯视，抛物/受力=侧视）

### 验证
- 新增 `backend/test_new_scene.py`，3 项场景选择测试全部通过：
  - 3D 抛物 → `projectile_motion` ✅
  - 3D 受力 → `force_analysis` ✅
  - 2D 抛物 → `projectile_motion_2d` ✅

## 2026-07-26 — 产品定位确认：聚焦数学物理可视化

### 定位明确
- **当前阶段**：聚焦简单数学题和物理题的可视化
  - 数学：几何证明、函数图像变换、方程可视化等
  - 物理：抛物运动、力学分析、电路演示等
- **后续扩展**：技术文档可视化、互动游戏（闯关问答、角色扮演）
- **渐进路线**：数学物理题 → 技术文档 → 互动游戏 → 全学科覆盖

### 产品核心价值
- **可视化**：将抽象公式和定理变成可看的 3D/2D 动画
- **可交互**：拖拽旋转视角、实时调整参数、分步导航
- **探究式学习**：学生可以改变题目参数，即时观察结果变化（如"角度变大，球会飞得更远吗？"）

### 比赛文档
- 开始填写 2026 中国高校计算机大赛人工智能创意赛（鸿蒙赛道）作品说明文档
- 赛题方向：Agent 创新
- 围绕数学物理可视化方向撰写介绍文档

## 2026-07-24 — 简化：两按钮 + 自动降级

### UI 精简
- 从四按钮简化为 **2D 动画 / 3D 动画** 两个按钮
- 每个按钮内部自动走"模板优先"策略，用户无需关心模板 vs AI

### 自动降级机制
- **后端**: mode=2d/3d 统一先尝试 lesson JSON 生成
  - 成功（scene.type 匹配具体模板）→ 返回 lesson_json
  - 失败（异常 / 仅匹配 generic）→ 自动调用 generate_animation_html() 生成 HTML
  - 降级时返回 `fallback: true` + `fallback_reason`，前端无感知
- **前端**: 统一检查 `data.lesson_json` 存在就用模板播放器，否则用 `data.animation_html` srcdoc 注入

### 移除的内容
- 移除 `template` / `template-2d` 独立 mode 值
- 前端四按钮缩减为两按钮
- server.py mode 校验简化为 `("2d", "3d")`

## 2026-07-24 — 新增 2D 模板驱动方案

### 2D Canvas 模板播放器
- **新建** `frontend/player-template-2d.html` — Canvas 2D 俯视视角播放器
  - 道路 + 小车（俯视，带方向指示） + LiDAR 同心圆波纹 + 障碍物 + 路径标记
  - 背景粒子系统增强画面氛围
  - 支持 postMessage 加载数据 + URL hash base64 调试模式
- **新建** `LESSON_SYSTEM_PROMPT_2D` + `generate_lesson_json_2d()` 于 `script_generator.py`
  - 场景类型：`lidar_obstacle_avoidance_2d`、`generic_2d`
  - 使用二维坐标 [x, y] 描述物体位置
- **新增** `mode=template-2d` API 端点 + `GET /player-template-2d.html` 路由
- **前端** 四按钮模式切换：模板 3D / 模板 2D / AI 2D / AI 3D

### 测试文件新增
- `backend/test_template_2d.py` — 2D lesson JSON 生成测试
- `backend/test_api_2d.py` — 2D 模板全流程 API 测试
- 两项测试均通过，耗时 ~95s（与 3D 模板一致）

## 2026-07-24 — 模板驱动架构：从"AI 生成代码"到"AI 生成数据 + 固定模板渲染"

### 核心架构转型
- **动机**: manual-agent-demo-v1 验证了"数据驱动 3D 教学动画"的可行性，且质量远高于 AI 直接生成 HTML
- **旧方案**: AI 每次从零写 Three.js/Cavnas 代码 → 质量波动大、耗时长（~190s）、偶尔逻辑错误
- **新方案**: AI 只输出结构化 lesson JSON（~2KB），前端用固定 Three.js 模板渲染 → 质量恒定、速度快（~40s）

### 新增文件
- `frontend/player-template.html` — 基于 Three.js 的 3D 模板播放器
  - 使用 importmap 加载 Three.js CDN (0.160.0) + OrbitControls
  - 场景类型注册表：`lidar_obstacle_avoidance`、`generic_3d`
  - 接收 postMessage 加载 lesson 数据，暴露 `window.changeScene(n)`
  - 支持 URL hash 传 base64 编码的 JSON（方便调试）
- `backend/test_template.py` — lesson JSON 生成测试脚本
- `backend/test_api_template.py` — 全流程 API 测试（template 模式）

### 修改文件
- `backend/script_generator.py` — 新增 `LESSON_SYSTEM_PROMPT` + `generate_lesson_json()`
  - 场景类型：lidar_obstacle_avoidance（检测→识别→响应）、generic_3d（兜底）
  - AI 被要求输出严格 JSON Schema，自动选择场景类型
  - 温度 0.5、max_tokens 4096、非流式（速度快）
- `backend/server.py` — `/api/generate?mode=template` 新增端点
  - mode 参数扩展为 "2d"/"3d"/"template"，默认改为 "template"
  - template 模式返回 `{scenes, lesson_json, mode}`（无 animation_html）
  - 新增 `GET /player-template.html` 路由
- `frontend/index.html` — 三种模式支持
  - 模式切换从拨动开关改为三个按钮（模板 3D / 2D / 3D）
  - 默认模式改为 "template"
  - 模板模式：iframe 加载 player-template.html → postMessage 传 lesson 数据
  - HTML 模式（2D/3D）：保留原有 srcdoc 注入方式作为兜底

### 性能对比
| | 旧方案（AI HTML） | 新方案（Template） |
|---|---|---|
| 动画生成耗时 | ~190s | **~40s** |
| 总流程耗时 | ~260s | **~95s** |
| 输出大小 | ~43KB HTML | ~2KB JSON |
| 渲染质量 | AI 波动，偶有逻辑错误 | 模板保证，100% 可靠 |

### manual-agent-demo-v1 同步改动
- 将硬编码 `steps` 抽成 `src/lesson.json`
- 重构 `src/App.tsx` 为完全数据驱动：从 lesson.json 读取 meta/scene/steps
- 组件拆分：Lidar、CarBody、CarWheels、CarTop、Obstacle、PathMarkers、CameraControls
- 每个组件接收 `state: ObjectState` 参数，由步骤数据控制动画

## 2026-07-24 — 2D/3D 双模式 + 动画质量优化

### 前端新增 2D/3D 切换
- **文件**: `backend/script_generator.py`, `backend/server.py`, `frontend/index.html`
- `ANIMATION_SYSTEM_PROMPT` 拆分为 `_2D` 和 `_3D` 两套独立 prompt
- `generate_animation_html(scenes, mode)` 新增 mode 参数
- `POST /api/generate?mode=2d|3d` 支持选择动画类型
- 前端新增滑动开关 UI（默认 3D），toggle 交互逻辑

### 3D Three.js prompt 打磨
- 将 Canvas 2D prompt 全面升级为 Three.js/WebGL
- 新增：场景搭建指南、摄像机运镜（俯视45°/顶视90°/侧面平视）
- 新增：灯光系统（AmbientLight + DirectionalLight + PointLight）
- 新增：材质指南（MeshStandardMaterial/MeshPhongMaterial）
- CDN 策略调整：仅允许 Three.js CDN (unpkg.com/three@0.160.0)

### 物体朝向细节规范
- 新增"第4节：物体朝向与运动方向必须一致"
- 车头必须有视觉标识（黄色前灯）
- 转向用 Math.atan2 + lerp 平滑旋转
- 弧线轨迹上每帧根据切线方向更新朝向
- 小车建模建议：车体+车顶+前灯+车轮的具体尺寸

### 动画逻辑与细节要求（prompt 强化）
- 新增"分阶段设计"要求（接近→探测→避障，每阶段有明确时长）
- 新增"逻辑正确性"红线：探测距离≥50px(2D)/≥2单位(3D)、禁止瞬移、禁止碰撞
- 新增"细节刻画"：雷达波半透明扩散、红色障碍物脉动、检测反馈闪烁
- 新增"动画设计前必问三问"：核心动作？最佳视角？（俯视/侧面/正面）观众能懂？

## 2026-07-24 — 从"文字展示"到"概念具象化"动画转型

### 动画 prompt 根本性重写
- **文件**: `backend/script_generator.py`
- 核心改变：禁止把知识点做成文字标签/气泡堆在画面上，用画面演示知识点的来龙去脉
- 举例：雷达避障画小车+雷达波+石头+转向，而不是"雷达"两个大字飘来飘去
- 画面上最多 1 行字幕 + 1 个标题，其余全用动画画面
- Canvas 2D 为主要实现方式

## 2026-07-23 — 用 DeepSeek 替代 Kimi 生成动画

### 动画模型切换
- **文件**: `backend/script_generator.py`
- `generate_animation_html()` 从 Kimi (kimi-k2.7-code) 切换为 DeepSeek (deepseek-v4-pro)
- 原因：Kimi 组织级并发限制(3)，经常 429 限流；DeepSeek 更稳定
- max_tokens: 12288 → 16384, timeout: 150s → 180s
- Doubao (豆包 Seedance) 方案尝试后放弃：Volcengine Ark 需要创建推理端点，API 不通

### 清理
- 删除 `DOUBAO_KEY/DOUBAO_BASE/DOUBAO_MODEL` 配置和 `_doubao` 客户端
- 删除 `test_seedance.py`、`test_deepseek_anim.py`

## 2026-07-23 — 稳定性修复（超时 + 限流 + 取消）

### 流式生成修复 ReadTimeout
- **文件**: `backend/script_generator.py`
- 动画生成改为 `stream=True`（之前阻塞等待全量响应导致 httpx.ReadTimeout）
- 流式生成 11000+ 片段, 43000+ 字符，耗时 ~190s

### 429 限流重试
- **文件**: `backend/script_generator.py`
- 新增 `_retry_on_rate_limit(factory, label)` 工厂函数
- 三个 API 调用全部包装：Kimi场景、DeepSeek优化、动画生成
- 遇到 429 等 3s 重试一次

### 前端取消逻辑修复
- **文件**: `frontend/index.html`
- 新增 `AbortController` + `abortedByTimeout` 标记
- 超时触发 abort → 保持错误卡片显示，不再自动跳回上传页
- 用户点取消 → 静默回到上传页
- TIMEOUT_MS: 180000 → 300000 → 600000 (10min)

### 服务端错误处理
- **文件**: `backend/server.py`
- 新增 `APITimeoutError` 捕获 → 504 + 中文提示
- 文件大小限制 10MB，格式校验 .pptx/.docx/.pdf

## 2026-07-23 — 初始实现

### 项目搭建
- **文件**: `backend/parser.py`, `backend/script_generator.py`, `backend/server.py`, `frontend/index.html`
- 三模型协作管道：Kimi(场景拆分) → DeepSeek(场景优化) → Kimi(动画HTML生成)
- FastAPI 单端点 `/api/generate`
- 前端三状态 SPA：Upload → Loading → Player (iframe + changeScene 导航)
- 文件解析：python-pptx / python-docx / pdfplumber
