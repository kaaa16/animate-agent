# 待办事项

> **产品定位**（2026-07-26 确认）：当前聚焦**简单数学题 + 物理题的可视化**，后续扩展至技术文档、互动游戏。渐进路线：数学物理 → 技术文档 → 游戏 → 全学科。

## 已知问题

### ~~生成时间长~~ → 已大幅改善
- 旧方案 ~260s，模板方案 ~95s（lesson JSON 生成仅 ~40s）
- 瓶颈已从动画生成转移到场景拆分（Kimi ~30s + 网络开销 ~25s）
- 进一步优化方向：场景拆分缓存、预热连接

### Kimi 并发限制
- Kimi 组织级并发限制为 3，多用户或多请求时会 429
- 已有重试机制（等 3s 重试一次），但重试后仍可能失败
- 长期方案：用 DeepSeek 替代 Kimi 做场景拆分

### 动画场景类型有限
- 仅支持 `lidar_obstacle_avoidance`（检测→识别→响应）和 `generic_3d`（兜底）
- AI 有时把不匹配的知识点强行映射到避障场景（如"变量赋值"→扫描内存环境）
- **急需为数学物理知识点新增专用场景模板**

### 路径 waypoints 质量
- AI 生成的 waypoints 有时是直线（不绕障），需 prompt 约束或后处理

### 交互细节（低优先级，暂缓）
- 2D 播放器缺拖拽平移（左右拉动画面）
- 3D 播放器缺平移（拖动整个模型移动，当前 OrbitControls `enablePan=false`）
- 参数面板数值框宽度随位数变化抖动（如 8 vs 8.5）

## 待实现功能

### 高优先级（围绕数学/物理可视化）

- [ ] **新增数学/物理场景模板**（最重要！）：
  - ✅ `projectile_motion` — 抛物运动（已实现，含 2D 版 `projectile_motion_2d`，含速度分解 + 读数 + 暂停）
  - ⏳ `projectile_horizontal` — 平抛运动（初速度水平；待做，需含与斜抛同款速度分解 + 滑杆）
  - `function_graph` — 函数图像（坐标轴、曲线绘制、平移/伸缩变换、切线跟随）
  - `geometry_proof` — 几何证明（三角形、辅助线、角度标注、全等/相似变换动画）
  - ✅ `force_analysis` — 受力分析（已实现，含 2D 版 `force_analysis_2d`；斜面/滑轮待扩展）
  - `circuit_demo` — 电路演示（电源、电阻、电流流动动画、电压可视化）
  - `wave_visualization` — 波的传播（横波/纵波、干涉叠加、驻波节点）
  - `collision_simulation` — 碰撞模拟（动量守恒、弹性/非弹性碰撞、能量条）
- [x] **参数交互**：学生在播放器中拖动滑块调节题目参数（初速度、角度、质量等），即时看到可视化结果变化（✅ 已实现 projectile + force_analysis，3D/2D）
- [ ] 场景匹配优化：prompt 中增加"不匹配时回退 generic"逻辑
- [ ] 动画缓存：相同题目+相同 mode 不重复生成
- [ ] 错误重试按钮

### 中优先级
- [ ] SSE/WebSocket 生成进度推送
- [ ] 支持 .txt / .md / 纯文本直接输入
- [ ] 动画导出 mp4
- [ ] 播放器中更多 3D 细节（物体运动轨迹拖尾、摄像机运镜等）

### 低优先级
- [ ] 上传历史、批量处理、主题切换、国际化
- [ ] 技术文档可视化（产品定位第二阶段）
- [ ] 互动游戏模块（闯关问答、角色扮演）

## 架构演进（IR 中间表示层）
> 目标流程：`DocumentIR → LessonIR → StoryboardIR → { SVG / Three.js / Manim }`，渲染器与故事板解耦

- [x] 定义 IR 契约：`backend/ir_schemas.py`（DocumentIR / LessonIR / StoryboardIR + 子模型）
- [x] Ingest → DocumentIR：`parser.py` 结构化（read_* 返回 DocumentIR）
- [x] Knowledge Agent → LessonIR：`generate_script` 返回 LessonIR
- [x] 下游函数接收 LessonIR/dict/list：`_to_scenes_list`
- [ ] **StoryboardIR 真正落地**：lesson JSON 从 `{meta,scene,steps}` 重构为 `scenes[{objects,steps,renderer_hint}]` + 迁移 Three.js 渲染器（改动最大）
- [ ] 新增 SVG 渲染器（取代 Canvas 2D）
- [ ] 新增 Manim 渲染器
- [ ] 移除 AI-HTML 兜底 + Canvas 2D 旧路径（`player-template-2d.html`、`generate_lesson_json_2d`、`*_2d` 场景类型）

## 测试文件清单
| 文件 | 用途 | 状态 |
|------|------|------|
| `backend/test_anim.py` | 单独测试 DeepSeek 动画 HTML 生成 | ✅ 可用 |
| `backend/test_e2e.py` | 端到端测试（提取→场景→动画HTML） | ✅ 可用 |
| `backend/test_new_prompt.py` | 雷达避障场景测试 | ✅ 可用 |
| `backend/test_template.py` | lesson JSON 生成测试 | ✅ 新增 |
| `backend/test_api_template.py` | 全流程 API 测试（template 模式） | ✅ 新增 |
| `backend/test_new_scene.py` | 物理场景类型选择测试（projectile/force） | ✅ 新增 |
