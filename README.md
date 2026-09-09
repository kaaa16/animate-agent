# Intuition Engine Agent

Intuition Engine Agent 是一个把文本、文档、PPT 或开源项目 introduction 转成 2D/3D 教学动画的原型项目。项目目标不是生成普通摘要，而是输出 Interactive Knowledge Movie：把抽象概念拆成可见对象、因果动画、教学分镜和可交互参数。

## 当前可展示成果

- AI 前端输入页：`frontend/demo/index.html`
  - 支持粘贴文本、拖拽/选择本地文件、载入示例 prompt。
  - 有旋转地球、星空背景和输入分析结果预览。
  - 静态页面可直接用浏览器打开。

- 旧版交互课件原型：`frontend/demo/app.js`、`frontend/demo/styles.css`
  - 保留了机器人避障、ROS Publisher/Subscriber、API 鉴权等教学动画逻辑。
  - 当前在首页中作为 legacy prototype 保留，后续可重新接入路由或 demo 入口。

- Python 动画 spec 对象模型：`src/animate_agent/`
  - `animation/elements.py` 定义机器人、雷达、障碍物、ROS 节点、Topic、Message、Timeline、Scene 等可复用对象。
  - `animation/templates.py` 可生成小车避障和 ROS Publisher/Subscriber 的结构化动画 spec。
  - `interaction/controls.py` 定义 slider、button、toggle、drag target 等互动控件。

- 产品与交互设计文档：`docs/`
  - `PROJECT_EXPECTATIONS.md`：项目愿景和质量标准。
  - `TECH_STACK.md`：技术栈分析。
  - `TODO.md`：MVP 开发计划。
  - `INTERACTION_FLOW.md` 与 `interaction-flow.drawio`：最终交互流程叙述与流程图源文件。

## 快速运行

Python 侧测试：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

前端静态 demo：

```text
frontend/demo/index.html
```

## 工程原则

- 前端不直接执行模型生成的任意代码，只消费受控 JSON spec。
- Agent 组合对象库和模板库，减少重复生成代码和 token 消耗。
- 教学页面必须用具象对象解释核心逻辑，不能只把文档变成摘要。
- 每次新增或修改功能后提交一次，保持提交历史清晰。

## 这是一个测试