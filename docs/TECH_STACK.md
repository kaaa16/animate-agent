# 可视化动画与互动 Demo 技术栈分析

## 1. 总体原则

本项目的核心亮点是把手册内容转成“能看懂、能互动、能实验”的教学页面。更准确的产品定位不是文档阅读器，而是 **Interactive Knowledge Movie（交互式知识电影）**。

目标体验不是“上传 PDF，然后输出总结”，而是：

1. 上传 ROS 手册或其他技术手册。
2. AI 分析章节结构。
3. 生成知识图谱。
4. 进入第一页交互课件。
5. 滚轮推进镜头。
6. Publisher 出现。
7. Topic 飞出去。
8. Subscriber 收到消息。
9. 消息颜色变化。
10. 点击 Topic 弹出源码。
11. 点击源码后，对应 ROS 节点高亮。
12. 点击节点播放控制动画。
13. 拖动机器人并实时模拟状态变化。

技术栈要服务这个目标：

- Agent 负责理解文档、生成教学分镜和动画规格。
- 前端负责稳定渲染动画规格，不直接运行模型生成的任意代码。
- 常见动画对象和互动控件必须模板化、对象化，避免每次重新生成。

## 2. 后端技术栈

- Python 3.11+
  - 作为 Agent 编排、文档解析、动画 spec 生成的主语言。

- FastAPI
  - 提供文档上传、解析任务、分镜生成、动画 spec 生成 API。
  - 适合后续接异步任务、流式状态和前端调试面板。

- Pydantic
  - 定义文档块、概念图谱、分镜、动画元素、互动控件的 schema。
  - 后续模型输出必须通过 schema 校验后再交给前端。

- 文档解析
  - PDF：`pypdf` 起步，复杂版可升级 `PyMuPDF` 或 `unstructured`。
  - HTML：`BeautifulSoup` 抽取正文、标题、表格。
  - Markdown：`markdown-it-py` 解析章节结构。

- Agent 编排
  - 建议拆成五步：文档解析、概念抽取、流程关系抽取、教学分镜生成、动画 spec 生成。
  - 每一步都保存中间结果，方便调试和人工修正。

## 3. 推荐前端技术栈

- 当前 demo：原生 HTML/CSS/JavaScript + Canvas
  - 适合快速验证体验，不需要构建环境。

- 页面框架：React + Next.js
  - 适合做文档上传、任务流、交互课件、分享页和多页面课程。
  - TypeScript 定义和后端一致的动画 spec 类型。

- 赛博朋克 UI：Tailwind CSS + shadcn/ui + Framer Motion
  - Tailwind 负责快速建立统一视觉语言。
  - shadcn/ui 负责高质量基础组件。
  - Framer Motion 负责 UI 级转场和微交互。

- 页面滚动叙事：GSAP + ScrollTrigger
  - 用滚轮推进镜头、章节、对象出现、消息飞行和节点高亮。
  - 适合做“知识电影”的时间线体验。

- 2D 动画
  - Canvas：适合机器人、传感器、流程、坐标、队列、状态机等高频动画。
  - SVG：适合可缩放流程图、网络拓扑、静态结构图。
  - 推荐先建设 Canvas 2D 渲染器，因为互动性能和绘制自由度更高。
  - Rive：适合做可交互的角色、按钮、状态反馈和精细 2D 动效。
  - Motion Canvas：适合 Web 端可程序化时间轴动画。
  - Manim：适合离线生成数学/几何类讲解动画。

- 3D 动画
  - Three.js 或 React Three Fiber：适合设备结构、空间传感器、机械臂、工厂场景。
  - 需要 glTF/GLB 模型资产或可程序化生成的基础几何体。
  - 粒子、激光雷达、点云：Three.js + GLSL Shader。

- 图谱/拓扑
  - React Flow：适合 ROS 节点、Topic、服务、Action、系统架构和调用链。

- 动画控制
  - 基础阶段：自研 timeline spec，支持 play、pause、reset、step。
  - 复杂阶段：可引入 XState 管理状态机动画。
  - AI 生成动画脚本时，应生成场景 DSL，描述对象、动作、镜头和时序，再由渲染器执行。

## 4. 动画 Spec 设计

Agent 不应该直接输出完整前端代码，而应输出类似下面的结构：

```json
{
  "title": "小车避障教学动画",
  "elements": [
    { "id": "car", "type": "robot_car", "x": 120, "y": 310, "heading": 0 },
    { "id": "lidar", "type": "lidar_sensor", "owner_id": "car", "radius": 150 },
    { "id": "obstacle-1", "type": "obstacle", "x": 410, "y": 300, "radius": 34 }
  ],
  "timeline": [
    {
      "id": "scan",
      "title": "雷达扫描",
      "narration": "雷达向前方扇形区域发射多条测距射线。",
      "focus_element_ids": ["lidar"]
    }
  ],
  "controls": [
    {
      "id": "safe-distance",
      "type": "slider",
      "label": "安全距离",
      "target_property": "scene.safe_distance"
    }
  ]
}
```

这样做的好处：

- 安全：前端不执行任意代码。
- 省 token：Agent 复用对象类型，不重复生成绘制逻辑。
- 可测试：schema 可以单元测试，渲染器可以快照测试。
- 可迭代：同一个 spec 可由 Canvas、SVG 或 Three.js 不同渲染器消费。

## 5. 需要沉淀的模板库

- ROS/机器人中间件
  - Node、Publisher、Subscriber、Topic、Message、Service、Action、TF、参数服务器、源码片段、运行时状态。

- 机器人/自动驾驶
  - 小车、车头方向、轮子、轨迹、雷达、相机、障碍物、安全区域、路径规划线。

- API/系统架构
  - 客户端、网关、服务、数据库、请求包、鉴权节点、限流器、重试队列。

- 状态机/控制逻辑
  - 状态节点、转移箭头、条件、事件、当前状态高亮。

- 数据结构/算法
  - 队列、栈、树、图、指针、排序条、搜索边界。

- 工业设备/硬件
  - 传感器、电机、控制器、信号线、阈值区域、报警状态。

## 6. 额外配置需求

- 2D 教学动画：无需额外系统配置，浏览器即可运行。
- PDF 解析：需要安装 Python 解析库，复杂 PDF 可能需要 OCR。
- URL 文档抓取：需要后端网络权限和网页清洗策略。
- 高质量图片/纹理：需要素材库或图片生成模型。
- 3D demo：需要 Three.js、模型资源、贴图资源和性能测试。
- 自动生成动画：需要严格 schema、对象白名单、渲染器沙箱。
