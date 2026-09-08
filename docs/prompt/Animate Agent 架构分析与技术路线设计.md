# Animate Agent 架构分析与技术路线设计

我正在开发一个名为 **“把世界画出来”** 的 Animate Agent。

产品理念：

> 直觉是人类认识世界的第一方式，所以我们热衷于可视化。

产品目标不是普通的“文档阅读器”或“文档摘要工具”，而是把枯燥、密集、难读的技术文档转换成简单易懂、有趣、可互动的教学网页。

我希望最终产品定位为：

**Interactive Knowledge Movie（交互式知识电影）**

用户上传官方文档、技术手册、论文或教程后，Animate Agent 应该输出：

1. 文档核心逻辑的结构化理解
2. 面向初学者重新组织后的教学顺序
3. 教学分镜 Storyboard
4. 具象动画演示
5. 可以调整参数的 Interactive Demo
6. 与动画同步的简短文字解释
7. 必要时允许用户继续向 Agent 提问

重点：

不要把产品理解成“LLM 给用户生成一篇摘要”。

它应该更接近：

**知识理解 → 教学导演 → 动画编排 → 浏览器实时演出**

---

# 一、先分析一个核心问题：这个产品到底是不是 Agent？

请分析 Animate Agent 应该如何划分：

## Agent 智能层

负责：

- Document Understanding
- Concept Extraction
- Relationship Extraction
- Knowledge Graph
- Teaching Planning
- Storyboard Planning
- Visualization Selection
- Interaction Planning
- Renderer Selection
- Tool Calling
- Result Validation / Retry

Agent 的职责应该是：

> 理解“应该画什么、为什么这样画、先画什么、后画什么”。

而不是：

> 每一次都从零写完整动画代码。

---

## Web Application / Runtime 层

负责：

- 页面布局
- 动画播放
- Timeline
- Play / Pause
- Seek
- Slider
- 参数交互
- SVG 渲染
- Canvas 渲染
- WebGL / 3D 渲染
- 用户操作反馈

因此整个产品可以看成：

```text
Animate Agent
    │
    │ 理解 / 决策 / 导演
    ▼
Visual Storyboard
    │
    │ 描述“演什么”
    ▼
Animation Runtime
    │
    │ 决定“怎么演”
    ▼
Interactive Knowledge Movie
```

请基于这个思路判断：

现有项目中哪些部分应该属于 Agent，哪些应该属于普通 Web 应用。

如果当前架构将两者耦合，请指出问题。

---

# 二、不要把 Manim 作为核心实时渲染方案

我目前原来的想法可能是：

```text
PDF
 ↓
LLM 1
 ↓
教学脚本
 ↓
LLM 2
 ↓
Manim Python
 ↓
运行代码
 ↓
Manim Render
 ↓
MP4
 ↓
Web 播放
```

请分析这个方案的问题。

重点考虑：

- 多次 LLM 调用延迟
- Token 成本
- Manim Python 代码生成稳定性
- 代码执行失败
- LaTeX 错误
- 对象布局错误
- 重新 Render 的成本
- 视频编码时间
- 用户修改参数之后是否需要重新渲染
- MP4 是否真正适合 Interactive Knowledge Movie

我的判断是：

Manim 可以保留，但是应该降级为：

**Optional Renderer / Specialized Renderer**

适合：

- 数学证明
- 微积分
- 线性代数
- 几何
- 需要高质量视频导出的场景

而不应该成为整个系统默认的实时动画 Runtime。

请验证这个判断。

---

# 三、重点设计 Visual Storyboard IR / Animation DSL

这是整个系统我认为最重要的一层。

不要让 LLM 直接输出：

- HTML
- React
- GSAP 代码
- Manim Python
- Three.js 代码

而应该让 LLM 输出结构化：

**Visual Storyboard IR**

例如：

```json
{
  "title": "TCP Three-Way Handshake",
  "teaching_goal": "理解 TCP 为什么需要三次握手",

  "scene_type": "network_sequence",

  "actors": [
    {
      "id": "client",
      "label": "Client"
    },
    {
      "id": "server",
      "label": "Server"
    }
  ],

  "timeline": [
    {
      "time": 0,
      "action": "appear",
      "target": "client"
    },
    {
      "time": 0,
      "action": "appear",
      "target": "server"
    },
    {
      "time": 1,
      "action": "send_packet",
      "from": "client",
      "to": "server",
      "label": "SYN"
    },
    {
      "time": 2,
      "action": "send_packet",
      "from": "server",
      "to": "client",
      "label": "SYN + ACK"
    },
    {
      "time": 3,
      "action": "send_packet",
      "from": "client",
      "to": "server",
      "label": "ACK"
    }
  ],

  "controls": [
    {
      "type": "slider",
      "parameter": "network_latency",
      "min": 10,
      "max": 1000,
      "default": 100,
      "unit": "ms"
    }
  ],

  "explanations": [
    {
      "time": 1,
      "text": "客户端首先确认自己能够向服务器发送数据。"
    }
  ]
}
```

然后前端 Animation Runtime 根据：

```text
scene_type = network_sequence
```

寻找：

```text
NetworkSequenceRenderer
```

而不是重新生成代码。

请重点设计这一层。

需要思考：

- JSON Schema
- TypeScript Types
- Zod Schema
- Scene
- Actor
- Object
- Action
- Timeline
- Transition
- Camera
- Interaction
- Parameter
- Explanation
- Simulation
- Formula
- Highlight
- Annotation

以及：

如何设计得既足够通用，又不至于成为一个极度复杂的“通用动画编程语言”。

---

# 四、设计 Knowledge Visualization Component Library

我希望后期逐渐积累一套知识动画组件库。

例如：

```text
TCP
→ PacketFlowScene

HTTP
→ RequestResponseScene

状态机
→ StateMachineScene

排序算法
→ ArraySortScene

BFS / DFS
→ GraphTraversalScene

神经网络
→ NeuralNetworkScene

Transformer
→ TokenFlowScene

数据库 B+ Tree
→ TreeScene

CPU Pipeline
→ PipelineScene

PID
→ ControlLoopScene

傅里叶变换
→ WaveDecompositionScene

SLAM
→ RobotMappingScene

激光雷达
→ LidarScanScene

坐标变换
→ CoordinateTransformScene

光学反射
→ RayOpticsScene
```

Animate Agent 的重要能力应该变成：

```text
Knowledge
 ↓
理解知识结构
 ↓
选择 Visualization Pattern
 ↓
生成对应 Scene 参数
 ↓
Animation Runtime
```

而不是：

```text
Knowledge
 ↓
LLM 从零开始写动画
```

请帮我设计：

```text
Renderer Registry
Scene Registry
Visualization Registry
```

例如：

```typescript
const rendererRegistry = {
  packet_flow: PacketFlowScene,
  state_machine: StateMachineScene,
  graph_traversal: GraphTraversalScene,
  array_sort: ArraySortScene,
  coordinate_system: CoordinateSystemScene,
  ray_optics: RayOpticsScene,
  robot_navigation: RobotNavigationScene
}
```

并分析如何让 Agent 根据知识类型选择合适 Renderer。

---

# 五、前端动画技术路线分析

优先考虑浏览器实时生成，而不是服务器提前 Render 视频。

目前候选技术：

## React + TypeScript

负责：

- 页面
- Component
- State
- UI
- Interaction

---

## SVG

重点用于：

- 技术图
- 箭头
- 节点
- 流程
- 几何
- 网络协议
- 算法动画

---

## GSAP

用于：

- Timeline
- Tween
- Seek
- Play
- Pause
- Object movement
- Path Animation
- SVG Animation

Interactive Knowledge Movie 很重要的一点是：

用户拖动：

```text
──────●────────
```

Animation Timeline 应该可以：

```text
seek(time)
```

而不是重新生成动画。

请分析 GSAP 是否适合作为 Timeline Engine。

---

## D3

用于：

- 图表
- 数据
- 曲线
- 坐标系统
- 树
- 图
- 算法可视化

---

## Canvas

用于：

- 大量对象
- 粒子
- 高频动画

---

## Three.js / React Three Fiber

用于：

- SLAM
- Robot
- 机械臂
- LiDAR
- 点云
- 3D 坐标系
- 空间几何
- 物理现象

---

## Motion Canvas

请重点分析是否适合 Animate Agent。

我希望了解它是否适合：

- TypeScript 动画
- 教学动画
- Timeline
- Scene
- 实时 Preview

以及是否值得作为 Animation Runtime 的一部分。

---

## Manim

只作为：

```text
Special Renderer
```

---

## Remotion

考虑作为：

```text
Export Renderer
```

即：

```text
Interactive Web
      ↓
Export
      ↓
MP4
```

请分析这种设计。

---

# 六、设计 Renderer Router

目标架构：

```text
                    Storyboard

                        │
                        ▼

                 Renderer Router

        ┌───────────────┼───────────────┐

        ▼               ▼               ▼

     SVG/GSAP           D3            Three.js

        │               │               │

        ▼               ▼               ▼

       Web             Web             Web


                        │

                  特殊情况下

                        ▼

                      Manim

                        │

                        ▼

                      Video
```

例如 Agent 判断：

```text
TCP 三次握手
→ SVG + GSAP

快速排序
→ SVG / D3

Loss Curve
→ D3

SLAM
→ Three.js

机械臂
→ Three.js

傅里叶级数
→ SVG / Manim

数学证明
→ Manim
```

请设计 Renderer Router 的架构。

重点回答：

Agent 应该基于什么信息选择 Renderer？

例如：

```text
concept_type
visualization_type
dimension
object_count
requires_simulation
requires_math
requires_3d
requires_high_quality_video
requires_interaction
```

---

# 七、减少 LLM 调用次数

我不希望流程变成：

```text
每个 Scene 调一次 LLM

每个动画调一次 LLM

每次修改参数又调一次 LLM
```

目标是：

## 第一次 LLM

```text
Document
   ↓
Knowledge Model
```

输出：

```json
{
  "concepts": [],
  "relationships": [],
  "learning_objectives": [],
  "prerequisites": [],
  "difficulty": ""
}
```

## 第二次 LLM

```text
Knowledge Model
       ↓
Storyboard Planner
       ↓
Visual Storyboard JSON
```

之后动画播放原则上：

**0 次 LLM。**

例如：

```text
slider
 ↓
JavaScript
 ↓
Physics / Math calculation
 ↓
Update Scene
```

而不是：

```text
slider
 ↓
LLM
 ↓
重新生成
```

请评估这个设计。

如果有些场景仍然需要 Agent 动态参与，也请指出应该在哪些情况下调用 LLM。

---

# 八、请分析现有项目代码

接下来请检查当前项目代码。

重点查找：

1. 当前前端技术栈
2. React / Vue / Next.js / Vite 使用情况
3. 动画目前怎么实现
4. 是否已经存在 GSAP / D3 / Three.js / Canvas
5. LLM 调用位置
6. Prompt 组织方式
7. 文档解析流程
8. 是否已经有 Storyboard / Scene 等概念
9. 是否直接让 LLM 生成 HTML / React
10. 是否已经集成 Manim
11. 前后端的数据结构
12. Agent 是否已经有 Tool Calling
13. Agent 是否拥有 Planning
14. 是否存在 Renderer 抽象层
15. 是否存在通用动画组件

不要先大规模修改代码。

先分析架构。

---

# 九、给出目标系统架构

希望最终类似：

```text
                        User

                          │

                    Upload Document

                          │

                          ▼

              Document Processing Layer

                          │

                          ▼

               Knowledge Understanding

                          │

                          ▼

                   Knowledge Model

                          │

                          ▼

                  Teaching Planner

                          │

                          ▼

                 Storyboard Planner

                          │

                          ▼

                Visual Storyboard IR

                          │

                          ▼

                  Renderer Router

           ┌──────────────┼──────────────┐

           ▼              ▼              ▼

       SVG + GSAP        D3        Three.js / R3F

           │              │              │

           └──────────────┼──────────────┘

                          ▼

                 Animation Runtime

                          │

                          ▼

           Interactive Knowledge Movie

                          │

             ┌────────────┴────────────┐

             ▼                         ▼

        User Interaction           Ask Agent

             │                         │

             ▼                         ▼

      Local Simulation            LLM / Tools


Optional:

Storyboard
     │
     ▼
Manim / Remotion
     │
     ▼
MP4 Export
```

---

# 十、请最终输出以下结果

请不要直接开始重构。

先输出一份架构分析报告，至少包含：

## 1. Current Architecture

根据项目代码说明当前架构。

---

## 2. Existing Problems

指出当前架构与上述 Animate Agent 理念之间的差距。

---

## 3. Target Architecture

设计新的 Animate Agent 架构。

---

## 4. Agent Boundary

明确哪些属于 Agent：

```text
Understanding
Planning
Storyboard
Renderer Selection
Tool Calling
Validation
```

哪些属于普通 Application：

```text
Rendering
Timeline
UI
Interaction
Simulation
```

---

## 5. Visual Storyboard IR

给出第一版 JSON Schema / TypeScript Interface 设计。

---

## 6. Renderer Architecture

设计：

```text
Renderer
RendererRegistry
RendererRouter
Scene
SceneComponent
TimelineEngine
InteractionEngine
SimulationEngine
```

---

## 7. Frontend Technology Recommendation

针对当前项目评估：

```text
React
SVG
GSAP
D3
Canvas
Three.js
React Three Fiber
Motion Canvas
Manim
Remotion
```

不要只解释这些技术是什么。

要说明：

**在 Animate Agent 中分别负责什么。**

---

## 8. LLM Pipeline

设计尽可能低延迟、低成本的 LLM 调用方式。

重点避免：

```text
LLM
→ code
→ execute
→ fail
→ LLM
→ retry
```

---

## 9. Migration Plan

从当前项目逐步迁移，不要推倒重写。

例如：

```text
Phase 1
定义 Storyboard IR

Phase 2
建立 Renderer Registry

Phase 3
实现 3~5 个通用 Scene

Phase 4
让 LLM 输出 Storyboard JSON

Phase 5
接入 GSAP Timeline

Phase 6
接入交互参数

Phase 7
Three.js Renderer

Phase 8
Manim / Remotion Export
```

---

## 10. MVP

请特别帮我压缩第一版 MVP。

第一版不要追求“所有知识都能画”。

先选择 5~10 种最有价值的 Visualization Pattern。

例如：

```text
Flow
Sequence
State Machine
Graph
Array
Chart
Coordinate System
Tree
Wave
Robot Map
```

分析哪些最值得优先开发。

---

# 核心设计原则

整个分析必须始终遵循以下原则：

### Principle 1

**LLM understands.**

大模型负责理解知识。

### Principle 2

**Agent directs.**

Agent 负责选择如何讲、如何画、如何互动。

### Principle 3

**Browser animates.**

浏览器负责实时动画，不依赖每次服务端 Render。

### Principle 4

**User explores.**

用户可以拖动、修改参数、暂停、回退、探索。

### Principle 5

**LLM should generate semantic instructions, not low-level animation code.**

LLM 应生成：

```text
Visual Storyboard IR
```

而不是每次生成：

```text
HTML
React
GSAP
Three.js
Manim
```

### Principle 6

Manim 是工具，而不是 Animate Agent 本身。

### Principle 7

真正长期有价值的资产应该是：

```text
Visual Storyboard DSL
+
Knowledge Visualization Component Library
+
Animation Runtime
+
Renderer Router
+
Teaching Planning Capability
```

---

最后，请站在“这是一个准备长期发展的产品，而不是一次 Demo”的角度，对当前项目进行架构分析。

优先考虑：

- 可扩展性
- 动画生成稳定性
- 用户即时反馈
- LLM 成本
- 延迟
- 交互能力
- 后期添加新知识动画组件的难度
- Agent 与 Web Runtime 的职责边界

先分析，再给方案，不要直接大规模改代码。