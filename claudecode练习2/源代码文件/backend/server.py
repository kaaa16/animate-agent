"""
FastAPI 服务器 — 教学动画生成器。
接收文件上传 → 提取文本 → AI 生成场景脚本 → AI 生成动画 HTML → 返回结果。
"""

import os
import sys
import traceback
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
sys.path.insert(0, BASE_DIR)

from openai import APITimeoutError
from parser import extract_text
from script_generator import generate_script, generate_animation_html, generate_lesson_json, generate_lesson_json_2d

app = FastAPI(title="教学动画生成器")

# CORS 允许所有来源（开发阶段）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pptx", ".docx", ".pdf"}


@app.get("/")
async def serve_index():
    """返回前端主页面"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>前端页面未找到</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/player-template.html")
async def serve_player_template():
    """返回 3D 模板播放器页面"""
    player_path = os.path.join(FRONTEND_DIR, "player-template.html")
    if not os.path.exists(player_path):
        return HTMLResponse(content="<h1>播放器页面未找到</h1>", status_code=404)
    with open(player_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/player-template-2d.html")
async def serve_player_template_2d():
    """返回 2D Canvas 模板播放器页面"""
    player_path = os.path.join(FRONTEND_DIR, "player-template-2d.html")
    if not os.path.exists(player_path):
        return HTMLResponse(content="<h1>播放器页面未找到</h1>", status_code=404)
    with open(player_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/api/generate")
async def api_generate(file: UploadFile = File(...), mode: str = "3d"):
    """
    文件上传 → AI 生成教学动画

    接收 PPT/Word/PDF 文件，提取文本后通过 Kimi + DeepSeek 双模型协作
    生成教学场景脚本和动画内容。

    参数:
        mode: "2d" | "3d"，默认 "3d"
              先尝试模板驱动（lesson JSON + 前端模板渲染），
              模板无法匹配时自动降级为 AI 直接生成 HTML。

    返回:
        {
            "scenes": [...],
            "lesson_json": {...},      // 模板成功时
            "animation_html": "...",   // 降级时
            "mode": "2d" | "3d",
            "fallback": true/false     // 是否触发了降级
        }
    """
    if mode not in ("2d", "3d"):
        mode = "3d"  # 默认 3D
    # ── 格式校验 ──
    ext = os.path.splitext(file.filename or "unknown.txt")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式「{ext}」。请上传 .pptx / .docx / .pdf 文件。",
        )

    # ── 读取内容 + 大小校验 ──
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f} MB），最大支持 10 MB。",
        )

    # ── 写入临时文件 ──
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── 1. 提取文本 ──
        text = extract_text(tmp_path)
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="文件中未提取到文字内容。请确认文件包含可读取的文字（非扫描图片）。",
            )

        # ── 2. 双模型生成场景 ──
        scenes = await generate_script(text)

        # ── 3. 模板优先：尝试 lesson JSON，失败则降级 AI HTML ──
        response_mode = mode  # 保持原始 mode
        try:
            if mode == "2d":
                lesson_json = await generate_lesson_json_2d(scenes, original_text=text)
            else:
                lesson_json = await generate_lesson_json(scenes, original_text=text)

            scene_type = lesson_json.get("scene", {}).get("type", "")
            if scene_type in ("generic_2d", "generic_3d"):
                raise ValueError("no_matching_template")

            # 模板匹配成功
            return JSONResponse(content={
                "scenes": scenes,
                "lesson_json": lesson_json,
                "mode": response_mode,
                "fallback": False,
            })
        except Exception as e:
            # 模板失败 → 自动降级为 AI 直接生成 HTML
            print(f"[Server] 🔄 模板未匹配({e})，降级为 AI 生成 HTML ({mode.upper()})...")
            animation_html = await generate_animation_html(scenes, mode=mode)
            return JSONResponse(content={
                "scenes": scenes,
                "animation_html": animation_html,
                "mode": response_mode,
                "fallback": True,
                "fallback_reason": "未找到合适的动画模板，已自动切换为 AI 生成",
            })

    except HTTPException:
        raise
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI 服务响应超时，请稍后重试。如果多次出现，可尝试上传较小的文件。",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"处理失败：{str(e)}。请稍后重试或联系管理员。",
        )
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
