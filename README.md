# Animate Agent

Animate Agent 是一个把官方文档、设备手册、API 手册转成可视化教学页面和互动动画 demo 的 Agent 项目。

项目当前处于原型阶段：已有一个静态前端 demo，展示“小车避障手册”和“ROS Publisher/Subscriber 手册”如何被转成教学分镜、Canvas 动画和可互动知识课件。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

静态 demo 可直接用浏览器打开：

```text
frontend/demo/index.html
```

## 项目结构

```text
animate-agent/
  config/                 # 环境配置与运行配置
  data/
    samples/              # 示例文档、手册片段
    generated/            # 生成结果，本地运行产物
  docs/                   # 产品、架构、技术栈、Todo 文档
  frontend/
    demo/                 # 当前静态交互原型
  src/
    animate_agent/
      animation/          # 可复用动画对象、场景模板
      documents/          # 文档解析与结构化抽取
      interaction/        # 可复用互动控件对象
      rendering/          # 前端渲染 spec 生成
      storyboard/         # 教学分镜生成
  tests/                  # 单元测试与集成测试
```

## 当前重点

- 先用 Python 后端建立文档解析、Agent 编排、动画 spec 生成能力。
- 前端不要直接执行模型生成的任意代码，而是消费受控的动画 JSON spec。
- 常见动画元素和互动控件要封装成对象，Agent 只组合对象，减少重复生成代码和 token 消耗。
- 教学页面必须用具象对象解释核心逻辑，不能只把文档变成摘要。
- 产品体验定位为 Interactive Knowledge Movie（交互式知识电影），不是普通文档阅读器。
- 当前 ROS demo 已包含滚轮推进镜头、Topic 消息飞行、源码弹窗、节点高亮和拖动机器人模拟。

## 相关文档

- [项目期望](docs/PROJECT_EXPECTATIONS.md)
- [技术栈分析](docs/TECH_STACK.md)
- [开发 Todo](docs/TODO.md)
