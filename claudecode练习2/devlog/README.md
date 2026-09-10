# 教学动画生成器 — 项目概览

## 是什么
上传 PPT/Word/PDF 文件 → AI 自动生成卡通风格教学动画 HTML（2D Canvas 或 3D Three.js）。

## 如何运行

### 安装依赖
```bash
cd backend
pip install -r requirements.txt
```

### 启动后端
```bash
cd backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### 访问前端
浏览器打开 http://localhost:8000 （服务器直接托管前端页面）

## 技术栈
- **后端**: Python 3.12 + FastAPI + uvicorn
- **前端**: 原生 HTML/CSS/JS（单文件 SPA，三状态切换）
- **AI 模型**: Kimi (Moonshot API) + DeepSeek API
- **文档解析**: python-pptx, python-docx, pdfplumber
- **动画渲染**: Canvas 2D 或 Three.js (CDN)

## 文件结构
```
├── backend/
│   ├── server.py              # FastAPI 服务（GET /, GET /player-template.html,
│   │                          #   GET /api/health, POST /api/generate?mode=template|2d|3d）
│   ├── script_generator.py    # AI 调用：场景拆分( Kimi) → 场景优化(DeepSeek)
│   │                          #   → lesson JSON 生成(DeepSeek) / HTML 生成(DeepSeek)
│   ├── parser.py              # PPTX/DOCX/PDF 文本提取（结构化 DocumentIR）
│   ├── ir_schemas.py          # IR 中间表示层（DocumentIR/LessonIR/StoryboardIR）
│   ├── requirements.txt       # Python 依赖
│   ├── test_anim.py           # 单独测试动画 HTML 生成
│   ├── test_e2e.py            # 端到端测试（HTML 模式）
│   ├── test_template.py       # lesson JSON 生成测试
│   ├── test_api_template.py   # 全流程 API 测试（template 模式）
│   ├── test_new_prompt.py     # 雷达避障场景 prompt 测试
│   └── test_sample.pptx/docx  # 测试文件
├── frontend/
│   ├── index.html             # 主前端 SPA（上传→加载→播放三状态）
│   │                          #   支持 2D / 3D 两模式切换
│   ├── player-template.html   # Three.js 3D 模板播放器（数据驱动渲染）
│   └── player-template-2d.html  # Canvas 2D 模板播放器
└── devlog/                    # 开发者日志（本文件夹）
```

## 启动/停止命令
- 启动: `cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8000`
- 停止: 杀掉 python 进程（`Stop-Process -Name python`）
