# Intuition Engine Agent

Intuition Engine Agent 是一个把一段文本(之后会将文档、PPT或者某开源项目的introduction)转成2D/3D教学动画的 Agent 项目。


## 期望项目结构

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


## 相关文档

- [项目期望](docs/PROJECT_EXPECTATIONS.md)
- [技术栈分析](docs/TECH_STACK.md)
- [开发 Todo](docs/TODO.md)

## git管理
1. 每一次增加或者修改一个功能，就commit一次，保持提交历史清晰。
2. 如果是一个较大的功能修改，影响到已有版本的稳定，则向我申请是否新开一个分支开发。
