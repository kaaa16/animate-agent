"""
脚本生成器：将文本内容转化为分场景教学脚本 JSON。
调用 DeepSeek + Kimi 双模型协作（Kimi 理解内容，DeepSeek 结构化输出）。
"""

import os
import json
import asyncio
import re
from openai import AsyncOpenAI, RateLimitError

from ir_schemas import LessonIR

# ── API 配置 ──────────────────────────────────────────
# 密钥从环境变量读取（本地可放 backend/.env，该文件已被 .gitignore 忽略，不会上传）
def _load_env_file(path: str) -> None:
    """从 .env 文件加载 KEY=value（不覆盖已存在的环境变量）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

KIMI_KEY = os.environ.get("KIMI_KEY", "")
KIMI_BASE = "https://api.moonshot.cn/v1"
KIMI_MODEL = "kimi-k2.7-code"

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_MODEL_FLASH = "deepseek-v4-flash"

_kimi = AsyncOpenAI(api_key=KIMI_KEY, base_url=KIMI_BASE, timeout=120.0)
_deepseek = AsyncOpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE, timeout=120.0)

# ── 工具函数 ────────────────────────────────────────

async def _retry_on_rate_limit(factory, label: str = ""):
    """遇到 429 限流时等 3 秒重试一次。factory 是返回 coroutine 的可调用对象。"""
    try:
        return await factory()
    except RateLimitError:
        print(f"[{label}] 遇到限流(429)，等待 3 秒后重试...")
        await asyncio.sleep(3)
        return await factory()

# ── System Prompts ────────────────────────────────────
KIMI_SYSTEM = (
    "你是一位资深课程设计师。用户会给你一份文档的纯文本内容。"
    "请把内容拆分为 3~6 个教学场景，每个场景一小节。"
    "用 JSON 数组格式输出，每个元素包含：\n"
    '  "scene" (整数, 从1开始),\n'
    '  "title" (该场景标题),\n'
    '  "narration" (200字以内的讲解文案),\n'
    '  "key_points" (3个关键词的数组)\n'
    "直接输出 JSON 数组，不要包含任何其他文字，不要用 ```json 包裹。"
)

DEEPSEEK_SYSTEM = (
    "你是课程内容结构化专家。用户会给一份文档的纯文本内容，以及一份初步的场景拆分方案。"
    "请审查并优化这份方案，确保：每场景信息量适中、key_points 精准、narration 口语化。"
    "用 JSON 数组格式输出（与输入格式相同），直接输出 JSON 数组，不要包含任何其他文字。"
)


async def _call_kimi(text: str) -> list[dict]:
    """Kimi 第一轮：通读文本，拆分教学场景"""
    async def _call():
        return await _kimi.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": KIMI_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=1,
            max_tokens=4096,
        )
    resp = await _retry_on_rate_limit(_call, "Kimi场景")
    return _parse_json(resp.choices[0].message.content)


async def _call_deepseek(text: str, kimi_result: list[dict]) -> list[dict]:
    """DeepSeek 第二轮：审查优化 Kimi 的场景拆分"""
    prompt = (
        f"=== 原始文档内容 ===\n{text}\n\n"
        f"=== 初步场景拆分 ===\n{json.dumps(kimi_result, ensure_ascii=False, indent=2)}"
    )
    async def _call():
        return await _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL_FLASH,
            messages=[
                {"role": "system", "content": DEEPSEEK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
    resp = await _retry_on_rate_limit(_call, "DeepSeek优化")
    return _parse_json(resp.choices[0].message.content)


def _parse_json(raw: str) -> list[dict]:
    """尽力从 LLM 回复中提取 JSON 数组"""
    text = raw.strip()
    # 去掉可能的 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


async def generate_script(text: str) -> LessonIR:
    """
    双模型协作生成教学脚本。
    1. Kimi 通读文档 → 拆分场景
    2. DeepSeek 审查优化 → 输出最终 JSON

    返回格式：
    [
      {
        "scene": 1,
        "title": "变量是什么",
        "narration": "在编程世界里，变量就像是...",
        "key_points": ["变量", "赋值", "数据类型"]
      },
      ...
    ]
    """
    print("[Kimi]   正在分析文档内容，拆解教学场景...")
    kimi_result = await _call_kimi(text)
    scene_count = len(kimi_result) if isinstance(kimi_result, list) else 0
    print(f"[Kimi]   完成 — 拆分为 {scene_count} 个场景")

    print("[DeepSeek] 正在审查并优化场景脚本...")
    final = await _call_deepseek(text, kimi_result)
    print(f"[DeepSeek] 完成 — 最终输出 {len(final)} 个场景")

    return LessonIR(scenes=_clean_result(final))


def _clean_result(data: list[dict]) -> list[dict]:
    """确保每个元素的字段完整、类型正确"""
    cleaned = []
    for item in data:
        cleaned.append({
            "scene": int(item.get("scene", len(cleaned) + 1)),
            "title": str(item.get("title", "")),
            "narration": str(item.get("narration", "")),
            "key_points": [str(k) for k in item.get("key_points", [])[:3]],
        })
    return cleaned


def _to_scenes_list(scenes) -> list[dict]:
    """把 LessonIR / dict（含 'scenes' 键）/ list[dict] 统一转成 list[dict]。"""
    if isinstance(scenes, LessonIR):
        return scenes.model_dump()["scenes"]
    if isinstance(scenes, dict) and "scenes" in scenes:
        return scenes["scenes"]
    return scenes


# ── 动画 HTML 生成 ─────────────────────────────────

ANIMATION_SYSTEM_PROMPT_2D = (
    "你是一位资深创意动画设计师，精通 HTML/CSS/JS 动画和 Canvas 2D 开发。\n"
    "用户会给你一个教学场景的 JSON 数组，每个场景包含 scene（序号）、title（标题）、"
    "narration（讲解文案，约200字）、key_points（3个关键词）。\n\n"
    "## 核心原则：用画面讲故事，不要用文字堆砌\n"
    "你的任务是把抽象知识点变成具象的动画画面，让观众一看就懂。\n"
    "每个场景都必须是**演示知识点的微型动画**，而不是\"文字+动态背景\"的幻灯片。\n\n"
    "### 什么是\"用画面讲故事\"？举例说明：\n"
    "- 知识点\"雷达检测障碍物\"→ 俯视视角：画面中央一辆小车向前行驶，车身周围发出同心圆波纹（雷达信号），前方出现石头，雷达波碰到石头后高亮反射，小车检测到后自动计算新路线并转向绕开。\n"
    "- 知识点\"变量存储数据\"→ 画一个贴了\"x\"标签的盒子，数字10飞入盒子，后面又飞出来变成计算结果\n"
    "- 知识点\"光合作用\"→ 阳光粒子洒向绿叶，CO₂小球从左侧进入叶片，O₂小球从右侧冒出\n"
    "- 知识点\"循环结构\"→ 一个角色反复走过同一段路，每次带着不同的东西\n\n"
    "### 动画设计前必须思考（非常重要！）\n"
    "在下笔写代码之前，先问自己三个问题：\n"
    "1. 这个知识点最核心的\"动作\"或\"过程\"是什么？ → 把它作为动画的主线\n"
    "2. 从哪个视角看最清楚？\n"
    "   - 俯视（鸟瞰）：适合展示空间关系、运动轨迹、范围探测、路径规划、物体间相对位置\n"
    "   - 侧面：适合展示流程步骤、上下关系、层次结构、化学反应过程\n"
    "   - 正面：适合展示对话/交互、界面操作、前后对比\n"
    "3. 观众看完这个动画，能立刻理解知识点吗？ → 如果不能，简化画面，只保留最核心的元素\n\n"
    "### 硬性规则（违反会导致生成失败）\n"
    "- 禁止以大面积文字作为画面主体。narration 文字可浓缩为最多一行字幕出现在底部\n"
    "- 禁止把 key_points 做成文字标签/气泡/卡片堆在画面上\n"
    "- 画面上最多出现 1 行字幕 + 1 个标题标签，其余全部用动画画面来表达\n"
    "- 标题可以出现在画面顶部，但应该小巧精炼（不超过 8 个字），用淡入方式出现即可\n\n"
    "## 严格技术要求\n"
    "1. 输出完整的独立 HTML 文件（所有 CSS/JS 内嵌，不使用任何外部资源或 CDN）\n"
    "2. 深色主题，背景色使用 #0f0f1e\n"
    "3. 必须暴露全局函数：window.changeScene(n) — 参数 n 是从 0 开始的场景索引\n"
    "4. 页面加载时默认显示场景 0\n"
    "5. 场景切换时有流畅的过渡动画（淡入淡出，约 0.5~0.8 秒）\n"
    "6. 使用 Canvas 2D API 绘制动画：几何角色、物体、运动轨迹、粒子效果等\n"
    "7. 所有文字使用中文字体：font-family: \"PingFang SC\", \"Microsoft YaHei\", sans-serif\n\n"
    "## 动画风格要求\n"
    "- 生动的卡通风格，像给小朋友看的科普动画短片，画面要饱满、活泼、有趣\n"
    "- 使用圆润的形状、柔和的阴影、明亮的强调色（主色 #7c5ce7 紫色，辅色 #00d4aa 青色）\n"
    "- 每个场景根据教学内容设计独特的视觉画面和角色，不要所有场景长得一样\n"
    "- 可以设计简单的几何形状角色（圆形/矩形组成的可爱小人、动物、机器人）来表演知识点\n"
    "- 添加漂浮粒子或星空背景作为氛围，但不能喧宾夺主\n"
    "- 动画要有明确的\"故事线\"：起因→过程→结果，让观众理解知识点的来龙去脉\n\n"
    "## 动画逻辑与细节要求（决定成败！）\n"
    "动画不是\"会动的画\"，而是\"会讲故事的画\"。每一个动作都必须符合现实逻辑：\n\n"
    "### 1. 分阶段设计（每个场景至少包含 3 个阶段）\n"
    "以\"雷达避障\"为例：\n"
    "  阶段① 接近（~1.5s）：小车从左侧驶入画面，前方远处有一个石头障碍物\n"
    "  阶段② 探测（~1.0s）：小车发出同心圆雷达波，波纹扩散碰到石头后高亮反弹\n"
    "  阶段③ 避障（~1.5s）：小车减速→车头随弧线切线方向旋转→沿弧线轨迹从石头旁边绕过去→车头回正后加速继续行驶\n"
    "每个阶段之间要有自然的过渡，不能所有动作同时发生，也不能瞬间完成。\n\n"
    "### 2. 逻辑正确性（绝对不能出错！）\n"
    "- 探测器/传感器必须在接触障碍物**之前**就发现它（保持至少 50px 以上的探测距离）\n"
    "- 物体在转向/变向时必须沿弧线运动，不能瞬间改变方向或瞬移\n"
    "- 碰撞、反射、传递等物理动作要有明确的因果关系：A 触发 B，B 触发 C\n"
    "- 关键动作要有\"准备→执行→结果\"的完整过程，不能直接跳到结果\n"
    "- 如果动画中有\"避障\"或\"防止\"类动作，绝对不能让角色/物体撞上目标\n\n"
    "### 3. 细节刻画（2D Canvas 版本）\n"
    "- 雷达/信号/电波用半透明同心圆 + 虚线表示，从发射源向外扩散，扩散时要有淡出效果\n"
    "- 运动物体在改变方向前要先减速，转向时沿弧线轨迹运动，转向后逐渐加速\n"
    "- 障碍物/目标物体用醒目颜色（如 #ff6b6b 红色）并加发光或脉冲效果，让观众一眼看到\n"
    "- 被探测到的物体要有视觉反馈（高亮闪烁、变色、弹出感叹号等）\n"
    "- 关键动作发生时可以短暂暂停（约 0.3s），给观众理解的时间\n\n"
    "### 4. 物体朝向与运动方向必须一致（极其重要！）\n"
    "- 任何有\"前后\"概念的物体（车、角色、箭头等），其前方（车头/面部）必须始终朝向运动方向\n"
    "- 车头必须有明确视觉标识：用对比色方块做车头灯，或用三角形/箭头指示前方\n"
    "- 转向时使用 Math.atan2(dy, dx) 计算目标朝向角度，平滑过渡旋转，不能瞬间改变角度\n"
    "- 转弯过程：先减速 → 朝向缓慢旋转（角速度与转弯半径匹配）→ 朝向稳定后加速驶离\n"
    "- 物体在弧线轨迹上运动时，每一帧都要根据当前位置的切线方向更新朝向\n"
    "- 小车结构建议：车身主体矩形 + 车顶小矩形 + 车前两个黄色小方块（车灯）+ 四个圆形车轮。这样车的前后左右一目了然\n\n"
    "## Canvas 绘图指南\n"
    "- 使用 Canvas 2D API 绘制场景：ctx.fillRect、ctx.arc、ctx.beginPath 等\n"
    "- 使用 requestAnimationFrame 驱动动画循环\n"
    "- 每个场景应有独立的绘制函数，切换场景时清空 Canvas 并重新绘制\n"
    "- 动画要流畅（≥30fps），物体运动使用缓动函数（ease-in-out 等）\n"
    "- 粒子效果：用数组管理粒子，每帧更新位置和透明度后绘制小圆点\n\n"
    "## 禁止事项\n"
    "- 不要使用任何外部图片、字体或 CDN 资源\n"
    "- 不要使用 left/right 分栏布局\n"
    "- 不要包含任何导航按钮或页码（导航由父页面控制）\n"
    "- 不要输出 scene 编号或\"第X页\"之类的页码标记\n"
    "- 不要把 key_points 做成大字标题或卡片堆在画面中\n\n"
    "## 输出格式\n"
    "直接输出完整 HTML 代码。不要用 ```html 或任何代码块标记包裹。不要输出任何解释性文字。"
    "第一行必须是 <!DOCTYPE html>。"
)

ANIMATION_SYSTEM_PROMPT_3D = (
    "你是一位资深 3D 动画设计师，精通 Three.js 和 WebGL 开发。\n"
    "用户会给你一个教学场景的 JSON 数组，每个场景包含 scene（序号）、title（标题）、"
    "narration（讲解文案，约200字）、key_points（3个关键词）。\n\n"
    "## 核心原则：用画面讲故事，不要用文字堆砌\n"
    "你的任务是把抽象知识点变成具象的动画画面，让观众一看就懂。\n"
    "每个场景都必须是**演示知识点的微型动画**，而不是\"文字+动态背景\"的幻灯片。\n\n"
    "### 什么是\"用画面讲故事\"？举例说明：\n"
    "- 知识点\"雷达检测障碍物\"→ 俯视视角：画面中央一辆小车向前行驶，车身周围发出同心圆波纹（雷达信号），前方出现石头，雷达波碰到石头后高亮反射，小车检测到后自动计算新路线并转向绕开。俯视能清楚展示雷达覆盖范围、障碍物位置、避障路径三者的空间关系。\n"
    "- 知识点\"变量存储数据\"→ 画一个贴了\"x\"标签的盒子，数字10飞入盒子，后面又飞出来变成计算结果\n"
    "- 知识点\"光合作用\"→ 阳光粒子洒向绿叶，CO₂小球从左侧进入叶片，O₂小球从右侧冒出\n"
    "- 知识点\"循环结构\"→ 一个角色反复走过同一段路，每次带着不同的东西\n\n"
    "### 动画设计前必须思考（非常重要！）\n"
    "在下笔写代码之前，先问自己三个问题：\n"
    "1. 这个知识点最核心的\"动作\"或\"过程\"是什么？ → 把它作为动画的主线\n"
    "2. 从哪个视角看最清楚？\n"
    "   - 俯视（鸟瞰）：适合展示空间关系、运动轨迹、范围探测、路径规划、物体间相对位置\n"
    "   - 侧面：适合展示流程步骤、上下关系、层次结构、化学反应过程\n"
    "   - 正面：适合展示对话/交互、界面操作、前后对比\n"
    "3. 观众看完这个动画，能立刻理解知识点吗？ → 如果不能，简化画面，只保留最核心的元素\n\n"
    "### 硬性规则（违反会导致生成失败）\n"
    "- 禁止以大面积文字作为画面主体。narration 文字可浓缩为最多一行字幕出现在底部\n"
    "- 禁止把 key_points 做成文字标签/气泡/卡片堆在画面上\n"
    "- 画面上最多出现 1 行字幕 + 1 个标题标签，其余全部用动画画面来表达\n"
    "- 标题可以出现在画面顶部，但应该小巧精炼（不超过 8 个字），用淡入方式出现即可\n\n"
    "## 严格技术要求\n"
    "1. 输出完整的独立 HTML 文件（CSS/JS 内嵌，仅允许从 CDN 加载 Three.js：https://unpkg.com/three@0.160.0/build/three.min.js）\n"
    "2. 深色主题，3D 场景背景色使用 #0f0f1e\n"
    "3. 必须暴露全局函数：window.changeScene(n) — 参数 n 是从 0 开始的场景索引\n"
    "4. 页面加载时默认显示场景 0\n"
    "5. 场景切换时有流畅的过渡动画（淡入淡出，约 0.5~0.8 秒）\n"
    "6. 使用 Three.js 构建 3D 场景：几何体建模、材质着色、灯光系统、摄像机运镜\n"
    "7. 所有文字使用中文字体：font-family: \"PingFang SC\", \"Microsoft YaHei\", sans-serif\n\n"
    "## 动画风格要求\n"
    "- 生动的卡通风格，像给小朋友看的科普动画短片，画面要饱满、活泼、有趣\n"
    "- 使用圆润的形状、柔和的阴影、明亮的强调色（主色 #7c5ce7 紫色，辅色 #00d4aa 青色）\n"
    "- 每个场景根据教学内容设计独特的视觉画面和角色，不要所有场景长得一样\n"
    "- 可以设计简单的几何形状角色（圆形/矩形组成的可爱小人、动物、机器人）来表演知识点\n"
    "- 添加漂浮粒子或星空背景作为氛围，但不能喧宾夺主\n"
    "- 动画要有明确的\"故事线\"：起因→过程→结果，让观众理解知识点的来龙去脉\n\n"
    "## 动画逻辑与细节要求（决定成败！）\n"
    "动画不是\"会动的画\"，而是\"会讲故事的画\"。每一个动作都必须符合现实逻辑：\n\n"
    "### 1. 分阶段设计（每个场景至少包含 3 个阶段）\n"
    "以\"雷达避障\"为例：\n"
    "  阶段① 接近（~1.5s）：小车从左侧驶入画面，前方远处有一个石头障碍物\n"
    "  阶段② 探测（~1.0s）：小车发出同心圆雷达波，波纹扩散碰到石头后高亮反弹\n"
    "  阶段③ 避障（~1.5s）：小车减速→车头随弧线切线方向平滑旋转→沿弧线轨迹从石头旁边绕过去→车头回正后加速继续行驶\n"
    "每个阶段之间要有自然的过渡，不能所有动作同时发生，也不能瞬间完成。\n\n"
    "### 2. 逻辑正确性（绝对不能出错！）\n"
    "- 探测器/传感器必须在接触障碍物**之前**就发现它（在 3D 空间中保持至少 2 个单位以上的探测距离）\n"
    "- 物体在转向/变向时必须沿弧线运动，不能瞬间改变方向或瞬移\n"
    "- 碰撞、反射、传递等物理动作要有明确的因果关系：A 触发 B，B 触发 C\n"
    "- 关键动作要有\"准备→执行→结果\"的完整过程，不能直接跳到结果\n"
    "- 如果动画中有\"避障\"或\"防止\"类动作，绝对不能让角色/物体撞上目标\n\n"
    "### 3. 细节刻画（3D 版本）\n"
    "- 雷达/信号电波用半透明圆环（TorusGeometry + MeshBasicMaterial + transparent:true），从发射源向外扩散缩放，配合 opacity 淡出\n"
    "- 3D 物体的运动轨迹用虚线粒子尾迹（BufferGeometry + Points），物体转向时画出平滑贝塞尔弧线\n"
    "- 障碍物/目标物体用红色材质（#ff6b6b）+ 脉动缩放效果（sin 波驱动 scale），让观众一眼看到\n"
    "- 被探测到的物体要有视觉反馈：材质颜色闪烁（红→黄→白交替）、或整体 scale 脉动\n"
    "- 地面用浅灰色大平面，辅助展示物体的空间位置关系\n"
    "- 关键动作发生时可以短暂慢动作（deltaTime * 0.3），给观众理解的时间\n\n"
    "### 4. 物体朝向与运动方向必须一致（极其重要！）\n"
    "- 任何有\"前后\"概念的物体（车、角色、箭头等），其前方（车头/面部）必须始终朝向运动方向\n"
    "- 车头必须有明确视觉标识：用不同颜色的长方体（如黄色 BoxGeometry）做车头灯，或用锥形/三角形指示前方\n"
    "- 转向时使用 Math.atan2(dz, dx) 计算目标朝向角度，用 lerp/slerp 平滑旋转，不能瞬间改变角度\n"
    "- 转弯过程：先减速 → 朝向缓慢旋转（角速度与转弯半径匹配）→ 朝向稳定后加速驶离\n"
    "- 物体在弧线轨迹上运动时，每一帧都要根据当前位置的切线方向更新朝向（lookAt 或手动计算旋转）\n"
    "- 小车结构建议：车体=BoxGeometry(1.2,0.4,0.7)横向为主、车顶=略小的 BoxGeometry、前灯=两个小 BoxGeometry 黄色置于车头、车轮=四个 CylinderGeometry 置于两侧。这样车的前后左右一目了然\n\n"
    "## Three.js 3D 开发指南\n"
    "### 场景搭建\n"
    "- 使用 THREE.Scene + THREE.PerspectiveCamera + THREE.WebGLRenderer 构建场景\n"
    "- Renderer 设置：antialias: true, alpha: false, setPixelRatio(Math.min(window.devicePixelRatio, 2))\n"
    "- 每个场景一个 Group，切换场景时移除旧 Group 添加新 Group（或显示/隐藏）\n"
    "- 物体用基本几何体拼合：BoxGeometry（方块）、SphereGeometry（球）、CylinderGeometry（柱）、ConeGeometry（锥）、TorusGeometry（环）\n\n"
    "### 摄像机运镜\n"
    "根据知识点的最佳观看角度选择初始机位，并在动画中微调：\n"
    "- 俯视 45°（isometric 风格）：camera.position.set(8, 8, 8), camera.lookAt(0, 0, 0) — 最适合展示空间关系和运动轨迹\n"
    "- 顶视 90°：camera.position.set(0, 10, 0.1), camera.lookAt(0, 0, 0) — 最适合展示雷达范围、路径规划\n"
    "- 侧面平视：camera.position.set(0, 2, 10), camera.lookAt(0, 0, 0) — 最适合展示流程、层次结构\n"
    "- 动画过程中摄像机可以做缓慢的微旋转（±15°）或拉近拉远，增强 3D 感，但不能剧烈晃动\n\n"
    "### 灯光与材质\n"
    "- 必须使用至少两盏灯：AmbientLight(0x404060, 2) + DirectionalLight(0xffffff, 3)，方向光从右上方照下\n"
    "- 可选加 PointLight 作为特效光（如雷达波中心发光）\n"
    "- 材质用 MeshStandardMaterial 或 MeshPhongMaterial，给不同物体不同颜色：\n"
    "  主体角色用主色 #7c5ce7（紫色），关键目标用 #ff6b6b（红色），辅助元素用 #00d4aa（青色）\n"
    "- 地面用大平面 PlaneGeometry + MeshStandardMaterial，颜色略深于背景\n\n"
    "### 动画循环\n"
    "- 使用 requestAnimationFrame 驱动渲染循环，确保 ≥30fps\n"
    "- 用时钟（THREE.Clock）获取 delta time 做帧率无关的动画\n"
    "- 物体运动使用缓动函数（ease-in-out），不能线性匀速\n"
    "- 粒子效果用 BufferGeometry + Points 实现\n\n"
    "## 禁止事项\n"
    "- 不要使用除 Three.js 以外的任何外部图片、字体或 CDN 资源\n"
    "- 不要使用 left/right 分栏布局\n"
    "- 不要包含任何导航按钮或页码（导航由父页面控制）\n"
    "- 不要输出 scene 编号或\"第X页\"之类的页码标记\n"
    "- 不要把 key_points 做成大字标题或卡片堆在画面中\n\n"
    "## 输出格式\n"
    "直接输出完整 HTML 代码。不要用 ```html 或任何代码块标记包裹。不要输出任何解释性文字。"
    "第一行必须是 <!DOCTYPE html>。"
)

async def generate_animation_html(scenes, mode: str = "3d") -> str:
    """
    调用 DeepSeek 生成卡通风格的教学动画 HTML。
    生成的 HTML 暴露 window.changeScene(n) 供父页面导航。

    参数:
        scenes: 教学场景列表，每场景含 scene/title/narration/key_points
        mode: "2d" 使用 Canvas 2D, "3d" 使用 Three.js

    返回:
        完整的 HTML 字符串
    """
    scenes = _to_scenes_list(scenes)

    if mode == "2d":
        system_prompt = ANIMATION_SYSTEM_PROMPT_2D
        label = "DeepSeek动画(2D)"
    else:
        system_prompt = ANIMATION_SYSTEM_PROMPT_3D
        label = "DeepSeek动画(3D)"

    user_prompt = json.dumps(scenes, ensure_ascii=False, indent=2)

    print(f"[{label}] 开始流式生成动画 HTML...")
    async def _call():
        return await _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL_FLASH,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1,
            max_tokens=16384,
            timeout=180.0,
            stream=True,
        )
    stream = await _retry_on_rate_limit(_call, label)

    chunks: list[str] = []
    chunk_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)
            chunk_count += 1

    html = "".join(chunks).strip()
    print(f"[{label}] 流式生成完成 — 共 {chunk_count} 个片段, {len(html)} 字符")

    # 清理可能的代码块标记
    if html.startswith("```"):
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)

    # 确保以 DOCTYPE 开头
    if not html.startswith("<!DOCTYPE html>") and not html.startswith("<html"):
        html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head><meta charset="UTF-8"></head>\n<body>\n' + html + '\n</body>\n</html>'

    return html


# ── Lesson JSON 生成（模板驱动方案） ─────────────────

LESSON_SYSTEM_PROMPT = (
    "你是一位资深 3D 教学动画设计师。用户会给你一份文档文本和一份教学场景 JSON 数组。\n"
    "你的任务是把教学内容转化为一个结构化的 lesson.json，供前端 3D 渲染引擎（Three.js）播放。\n\n"
    "## 核心原则\n"
    "用 3D 画面演示知识点的来龙去脉，让观众一看就懂。不要做文字幻灯片。\n"
    "无论什么主题，都设法用一个具体的\"可视化场景\"来表达。\n\n"
    "## 场景类型选择\n"
    "你必须根据知识点的性质，选择最合适的 scene.type：\n\n"
    "### 'lidar_obstacle_avoidance' — 障碍检测/避障类\n"
    "适合任何\"检测→识别→响应\"类的知识点，例如：\n"
    "- 雷达/LiDAR/声纳探测障碍物\n"
    "- 自动驾驶避障、机器人路径规划\n"
    "- 免疫系统检测病原体（可把白细胞比作小车，病毒比作障碍物）\n"
    "- 质检系统检测缺陷品（传感器扫描 → 发现瑕疵 → 剔除）\n"
    "- 任何\"传感器发现目标 → 触发动作\"的教学内容\n\n"
    "此场景默认包含：地面、绿色小车（带黄色前灯）、旋转 LiDAR 扫描锥、红色障碍物、蓝色绕行路径。\n"
    "你可以调整物体的位置、大小、颜色来适配不同主题。\n\n"
    "步骤设计建议（3步）：\n"
    "  Step 1 — 扫描/探测阶段：lidar 快速旋转、高透明度，障碍物静止\n"
    "  Step 2 — 识别/检测阶段：障碍物脉动变红、lidar 减速\n"
    "  Step 3 — 响应/行动阶段：显示蓝色路径标记\n\n"
    "### 'projectile_motion' — 抛体/抛物运动类\n"
    "适合\"物体在重力作用下沿抛物线运动\"的知识点，例如：\n"
    "- 斜抛/平抛运动、炮弹轨迹、篮球投篮、足球射门\n"
    "- 射程、最大高度、飞行时间等物理量的讲解\n\n"
    "此场景默认包含：地面、红色小球（抛体）、蓝色点状抛物线轨迹、橙色初速度箭头、\n"
    "黄色最高点标记、绿色射程标记。小球会沿抛物线循环飞行。\n\n"
    "scene 需提供 projectile 参数：\n"
    '  "projectile": { "initialSpeed": 5, "angle": 45, "gravity": 5, "launch": [0, 0.3, 0] }\n'
    "步骤设计建议（3步）：\n"
    "  Step 1 — 抛出：显示初速度箭头+轨迹，小球起飞\n"
    "  Step 2 — 最高点：高亮最高点标记\n"
    "  Step 3 — 落地：显示射程标记+完整轨迹\n\n"
    "objectStates: velocity{visible}、trajectory{visible,opacity}、apex{visible}、range{visible}\n\n"
    "### 'force_analysis' — 受力分析类\n"
    "适合\"分析物体所受各力\"的知识点，例如：\n"
    "- 重力、支持力、摩擦力、拉力的受力分析\n"
    "- 牛顿定律、力的平衡、自由体图（free-body diagram）\n\n"
    "此场景默认包含：地面、紫色方块、四个力矢量箭头（重力蓝色向下、支持力绿色向上、\n"
    "拉力橙色向右、摩擦力红色向左）。\n\n"
    "步骤设计建议（3步）：\n"
    "  Step 1 — 展示物体：仅显示方块\n"
    "  Step 2 — 垂直力：显示重力+支持力\n"
    "  Step 3 — 水平力：显示拉力+摩擦力（全部四个力）\n\n"
    "objectStates: forces{gravity,normal,applied,friction 各自 true/false}\n\n"
    "### 'generic_3d' — 通用 3D 场景\n"
    "当知识点不适合避障场景时使用。包含地面和基本灯光。\n"
    "适合自由摆放 3D 物体来表达任意概念。\n\n"
    "## JSON Schema（必须严格遵守）\n\n"
    "{\n"
    '  "meta": {\n'
    '    "title": "知识点标题（8-20字）",\n'
    '    "subject": "学科名称",\n'
    '    "eyebrow": "顶部小字描述（6-12字）"\n'
    '  },\n'
    '  "scene": {\n'
    '    "type": "lidar_obstacle_avoidance",\n'
    '    "ground": { "size": [8, 5], "color": "#f8fafc" },\n'
    '    "camera": { "position": [5.8, 4.2, 7], "fov": 55 },\n'
    '    "car": {\n'
    '      "position": [0, 0.34, 0],\n'
    '      "bodySize": [1.5, 0.55, 0.9],\n'
    '      "bodyColor": "#22c55e",\n'
    '      "wheels": [\n'
    '        { "position": [-0.48, -0.29, 0.48], "radius": 0.18, "width": 0.18 },\n'
    '        { "position": [0.48, -0.29, 0.48], "radius": 0.18, "width": 0.18 }\n'
    '      ],\n'
    '      "top": { "radius": 0.18, "height": 0.12, "color": "#0f766e" }\n'
    '    },\n'
    '    "lidar": {\n'
    '      "position": [0, 1.1, 0],\n'
    '      "color": "#38bdf8",\n'
    '      "coneRadius": 1.9,\n'
    '      "coneHeight": 3.3\n'
    '    },\n'
    '    "obstacle": {\n'
    '      "position": [2.2, 0.5, 0],\n'
    '      "size": [0.8, 1, 0.8],\n'
    '      "color": "#ef4444"\n'
    '    },\n'
    '    "path": {\n'
    '      "waypoints": [[-2.3, 0.04, 0], [-0.8, 0.04, 0], [0.6, 0.04, -1.2], [2.4, 0.04, -1.2]],\n'
    '      "color": "#2563eb",\n'
    '      "emissive": "#1d4ed8"\n'
    '    }\n'
    '  },\n'
    '  "steps": [\n'
    '    {\n'
    '      "id": "唯一ID（英文小写+下划线）",\n'
    '      "title": "步骤标题（6-15字）",\n'
    '      "description": "讲解文案（50-150字，口语化）",\n'
    '      "highlights": ["高亮的对象ID"],\n'
    '      "objectStates": {\n'
    '        "lidar": { "speed": 0.06, "opacity": 0.24 },\n'
    '        "obstacle": { "scale": 1.0, "emissive": "#000000" },\n'
    '        "path": { "visible": false }\n'
    '      }\n'
    '    },\n'
    '    ...共3个步骤\n'
    '  ]\n'
    '}\n\n'
    "### objectStates 参数说明（按场景类型不同，字段不同）\n"
    "lidar_obstacle_avoidance 场景：\n"
    "- lidar.speed: 扫描锥旋转速度（0.025=慢, 0.06=快）\n"
    "- lidar.opacity: 扫描锥透明度（0.12=淡, 0.24=浓）\n"
    "- obstacle.scale: 障碍物缩放（1.0=正常, 1.12=脉动放大）\n"
    "- obstacle.emissive: 障碍物自发光色（\"#000000\"=无, \"#7f1d1d\"=红色警报）\n"
    "- path.visible: 路径标记是否可见（true/false）\n\n"
    "projectile_motion 场景：\n"
    "- velocity.visible: 初速度箭头是否可见\n"
    "- trajectory.visible / trajectory.opacity: 轨迹可见性/透明度（0.2~0.8）\n"
    "- apex.visible: 最高点标记是否可见\n"
    "- range.visible: 射程标记是否可见\n\n"
    "force_analysis 场景：\n"
    "- forces.gravity / forces.normal / forces.applied / forces.friction: 各力箭头是否可见\n\n"
    "## 输出要求\n"
    "- 直接输出 JSON，不要包含 ```json 或任何代码块标记\n"
    "- 不要输出任何解释性文字\n"
    "- JSON 必须完整、合法（注意尾逗号和引号转义）\n"
    "- 场景中的 objects 位置/大小要合理，物体之间不要重叠或穿透\n"
    "- 保持俯视/等距视角（camera.position 的 y 坐标应大于 3）\n"
    "- steps 必须是恰好 3 个步骤，每个步骤对应一个教学阶段，objectStates 只包含该场景类型支持的字段\n"
)

LESSON_SYSTEM_PROMPT_GENERIC = (
    "你是 3D 教学动画设计师，使用通用 3D 场景模板。\n"
    "scene.type 设为 \"generic_3d\"。\n"
    "场景中只有地面，你需要在 steps 的 description 中用文字描述应该出现的 3D 物体。\n"
    "其余要求与主 prompt 相同：直接输出 JSON，3 个步骤，不要代码块标记。\n"
)

def _parse_lesson_json(raw: str):
    """解析 lesson JSON，支持代码块与首尾花括号提取；失败返回 None。"""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


async def _call_lesson_json(call_factory, label: str) -> dict:
    """调用 DeepSeek 生成 lesson JSON；输出截断或 JSON 解析失败自动重试（最多 3 次）。"""
    last_raw = ""
    for attempt in range(1, 4):
        resp = await _retry_on_rate_limit(call_factory, label)
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        raw = (msg.content or "").strip()
        last_raw = raw
        print(f"[{label}] 第{attempt}次生成完成 — {len(raw)} 字符, finish_reason={finish}")
        if finish == "length":
            print(f"[{label}] ⚠ 输出被截断（finish_reason=length），自动重试...")
            continue
        lesson = _parse_lesson_json(raw)
        if lesson is None:
            print(f"[{label}] ⚠ JSON 解析失败，自动重试...")
            continue
        return lesson
    raise ValueError(f"[{label}] 多次重试仍无法获得有效 lesson JSON，最后输出: {last_raw[:300]}...")


async def generate_lesson_json(scenes, original_text: str = "", mode: str = "3d") -> dict:
    """
    调用 DeepSeek 生成结构化 lesson JSON（供前端模板渲染）。

    参数:
        scenes: 教学场景列表（来自 generate_script 的输出）
        original_text: 原始文档文本（可选，帮助 AI 更好理解内容）
        mode: "3d" 使用 3D 场景模板

    返回:
        lesson JSON dict，包含 meta/scene/steps
    """
    scenes = _to_scenes_list(scenes)

    label = "DeepSeek Lesson JSON"

    # 构建用户输入：场景数据 + （可选）原始文本摘要
    if original_text:
        text_summary = original_text[:2000]  # 取前 2000 字符供 AI 参考主题
        user_prompt = (
            f"=== 文档内容（参考） ===\n{text_summary}\n\n"
            f"=== 教学场景 ===\n{json.dumps(scenes, ensure_ascii=False, indent=2)}\n\n"
            "请根据文档内容和场景信息，生成结构化的 lesson.json。"
        )
    else:
        user_prompt = (
            f"=== 教学场景 ===\n{json.dumps(scenes, ensure_ascii=False, indent=2)}\n\n"
            "请根据场景信息，生成结构化的 lesson.json。"
        )

    print(f"[{label}] 开始生成 lesson JSON...")
    async def _call():
        return await _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": LESSON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=16384,
            timeout=120.0,
        )
    lesson = await _call_lesson_json(_call, label)

    # 验证必要字段
    if "meta" not in lesson:
        lesson["meta"] = {"title": scenes[0].get("title", "教学动画"), "subject": "", "eyebrow": ""}
    if "scene" not in lesson:
        lesson["scene"] = {"type": "generic_3d", "ground": {"size": [8, 5], "color": "#f8fafc"}}
    if "steps" not in lesson or not isinstance(lesson["steps"], list):
        lesson["steps"] = []
        for i, s in enumerate(scenes):
            lesson["steps"].append({
                "id": f"step{i+1}",
                "title": s.get("title", f"步骤{i+1}"),
                "description": s.get("narration", ""),
                "highlights": [],
                "objectStates": {},
            })

    # 确保 scene.type 合法
    valid_3d_types = ("lidar_obstacle_avoidance", "projectile_motion", "force_analysis", "generic_3d")
    valid_2d_types = ("lidar_obstacle_avoidance_2d", "projectile_motion_2d", "force_analysis_2d", "generic_2d")
    if mode == "2d":
        if lesson["scene"]["type"] not in valid_2d_types:
            lesson["scene"]["type"] = "generic_2d"
    else:
        if lesson["scene"]["type"] not in valid_3d_types:
            lesson["scene"]["type"] = "generic_3d"

    return lesson


# ── 2D 模板 Lesson JSON 生成 prompt ─────────────────

LESSON_SYSTEM_PROMPT_2D = (
    "你是一位资深 2D 教学动画设计师，精通 Canvas 2D 教学动画（俯视/侧视等多视角）。\n"
    "用户会给你一份文档文本和一份教学场景 JSON 数组。\n"
    "你的任务是把教学内容转化为一个结构化的 lesson.json，供前端 Canvas 2D 渲染引擎播放。\n\n"
    "## 核心原则\n"
    "用 2D 动画演示知识点的来龙去脉，让观众一看就懂。\n"
    "动画必须展示知识点的实际过程，而不是展示文字说明。\n\n"
    "## 场景类型选择（必须根据知识点性质选择最匹配的具体场景，不要轻易用 generic）\n\n"
    "### 'lidar_obstacle_avoidance_2d' — 障碍检测/避障类（俯视 2D）\n"
    "适合任何\"检测→识别→响应\"类的知识点：\n"
    "- 雷达/LiDAR/声纳探测障碍物、自动驾驶避障\n"
    "- 免疫系统检测病原体（小车=白细胞，障碍物=病毒）\n"
    "- 质检系统检测缺陷品、网络防火墙检测攻击\n"
    "- 任何\"传感器发现→触发动作\"的教学内容\n\n"
    "此场景为俯视视角，包含：道路、绿色小车（带黄色前灯）、雷达波纹（蓝色同心圆）、\n"
    "红色障碍物、蓝色绕行路径标记。\n\n"
    "步骤设计建议（3步，对应检测→识别→响应）：\n"
    "  Step 1 — 扫描/接近：lidar 快速旋转+高透明度，小车接近障碍物\n"
    "  Step 2 — 检测/识别：障碍物脉动发红光，lidar 减速\n"
    "  Step 3 — 响应/行动：显示蓝色绕行路径\n\n"
    "### 'projectile_motion_2d' — 抛体/抛物运动类（侧视 2D）\n"
    "适合\"物体在重力作用下沿抛物线运动\"的知识点：斜抛/平抛、投篮射门轨迹、射程/最高点讲解。\n\n"
    "此场景包含：地面线、红色小球（抛体，沿抛物线循环飞行）、蓝色点状轨迹、\n"
    "橙色初速度箭头、黄色最高点标记、绿色射程标记。\n\n"
    "scene 需提供 projectile 参数（物理单位，非像素）：\n"
    '  "projectile": { "initialSpeed": 5, "angle": 45, "gravity": 5 }\n'
    "步骤设计建议（3步）：\n"
    "  Step 1 — 抛出：显示初速度箭头+轨迹，小球起飞\n"
    "  Step 2 — 最高点：高亮最高点标记\n"
    "  Step 3 — 落地：显示射程标记+完整轨迹\n\n"
    "objectStates: velocity{visible}、trajectory{visible}、apex{visible}、range{visible}\n\n"
    "### 'force_analysis_2d' — 受力分析类（正视/侧视 2D）\n"
    "适合\"分析物体所受各力\"的知识点：重力/支持力/摩擦力/拉力、牛顿定律、力的平衡。\n\n"
    "此场景包含：地面线、紫色方块、四个力矢量箭头（重力蓝向下、支持力绿向上、\n"
    "拉力橙向右、摩擦力红向左）。\n\n"
    "步骤设计建议（3步）：\n"
    "  Step 1 — 展示物体：仅显示方块\n"
    "  Step 2 — 垂直力：显示重力+支持力\n"
    "  Step 3 — 水平力：显示拉力+摩擦力（全部四个力）\n\n"
    "objectStates: forces{gravity,normal,applied,friction 各自 true/false}\n\n"
    "### 'generic_2d' — 通用 2D 场景（最后兜底，尽量不用）\n"
    "只有当知识点确实不适合以上任何具体场景时才使用。可自由放置 2D 图形元素。\n\n"
    "## JSON Schema（以下 lidar 仅为结构示例；scene.type 请按上文场景类型说明选择）\n\n"
    "{\n"
    '  "meta": { "title": "标题(8-20字)", "subject": "学科", "eyebrow": "小字描述(6-12字)" },\n'
    '  "scene": {\n'
    '    "type": "lidar_obstacle_avoidance_2d",\n'
    '    "background": "#0f0f1e",\n'
    '    "car": { "position": [200, 300], "speed": 0.3, "color": "#22c55e" },\n'
    '    "obstacle": { "position": [450, 300], "size": 50, "color": "#ef4444" },\n'
    '    "path": { "visible": false, "waypoints": [[200,300],[350,300],[450,240],[650,240]] }\n'
    '  },\n'
    '  "steps": [\n'
    '    {\n'
    '      "id": "唯一ID",\n'
    '      "title": "步骤标题(6-15字)",\n'
    '      "description": "讲解文案(50-150字，口语化)",\n'
    '      "highlights": ["高亮对象ID"],\n'
    '      "objectStates": {\n'
    '        "lidar": { "speed": 0.06, "opacity": 0.24 },\n'
    '        "obstacle": { "scale": 1.0, "emissive": "#000000" },\n'
    '        "path": { "visible": false }\n'
    '      }\n'
    '    },\n'
    '    ...共3个步骤\n'
    '  ]\n'
    '}\n\n'
    "### objectStates 参数说明（按场景类型不同，字段不同）\n"
    "lidar_obstacle_avoidance_2d 场景：\n"
    "- lidar.speed: 雷达波扩散速度 (0.025=慢, 0.06=快)\n"
    "- lidar.opacity: 雷达波透明度 (0.12=淡, 0.24=浓)\n"
    "- obstacle.scale: 缩放 (1.0=正常, 1.12=放大脉动)\n"
    "- obstacle.emissive: 自发光色 (\"#000000\"=无, \"#7f1d1d\"=红色警报)\n"
    "- path.visible: 路径标记可见 (true/false)\n\n"
    "projectile_motion_2d 场景：\n"
    "- velocity.visible: 初速度箭头是否可见 (true/false)\n"
    "- trajectory.visible / trajectory.opacity: 轨迹可见性/透明度 (0.2~0.8)\n"
    "- apex.visible: 最高点标记是否可见 (true/false)\n"
    "- range.visible: 射程标记是否可见 (true/false)\n\n"
    "force_analysis_2d 场景：\n"
    "- forces.gravity / forces.normal / forces.applied / forces.friction: 各力箭头是否可见 (true/false)\n\n"
    "## 输出要求\n"
    "- 直接输出 JSON，不要代码块标记\n"
    "- scene.type 使用 *_2d 后缀\n"
    "- 恰好 3 个步骤\n"
    "- waypoints 使用二维坐标 [[x, y], ...]\n"
)

async def generate_lesson_json_2d(scenes, original_text: str = "") -> dict:
    """
    调用 DeepSeek 生成 2D 结构化 lesson JSON（供前端 Canvas 2D 模板渲染）。
    参数和返回值同 generate_lesson_json()，但使用 2D 专用 prompt。
    """
    scenes = _to_scenes_list(scenes)

    label = "DeepSeek Lesson JSON(2D)"

    if original_text:
        text_summary = original_text[:2000]
        user_prompt = (
            f"=== 文档内容（参考） ===\n{text_summary}\n\n"
            f"=== 教学场景 ===\n{json.dumps(scenes, ensure_ascii=False, indent=2)}\n\n"
            "请生成 2D 结构化 lesson.json（避障用俯视、抛物/受力用侧视，按知识点选）。使用 *_2d 场景类型。"
        )
    else:
        user_prompt = (
            f"=== 教学场景 ===\n{json.dumps(scenes, ensure_ascii=False, indent=2)}\n\n"
            "请生成 2D 结构化 lesson.json（避障用俯视、抛物/受力用侧视，按知识点选）。使用 *_2d 场景类型。"
        )

    print(f"[{label}] 开始生成 2D lesson JSON...")
    async def _call():
        return await _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": LESSON_SYSTEM_PROMPT_2D},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=16384,
            timeout=120.0,
        )
    lesson = await _call_lesson_json(_call, label)

    # 验证 + 兜底
    if "meta" not in lesson:
        lesson["meta"] = {"title": scenes[0].get("title", "教学动画"), "subject": "", "eyebrow": ""}
    if "scene" not in lesson:
        lesson["scene"] = {"type": "generic_2d", "background": "#0f0f1e"}
    if "steps" not in lesson or not isinstance(lesson["steps"], list):
        lesson["steps"] = []
        for i, s in enumerate(scenes):
            lesson["steps"].append({
                "id": f"step{i+1}",
                "title": s.get("title", f"步骤{i+1}"),
                "description": s.get("narration", ""),
                "highlights": [],
                "objectStates": {},
            })

    valid_2d_types = ("lidar_obstacle_avoidance_2d", "projectile_motion_2d", "force_analysis_2d", "generic_2d")
    if lesson["scene"]["type"] not in valid_2d_types:
        lesson["scene"]["type"] = "generic_2d"

    return lesson


def generate_animation_html_sync(scenes: list[dict]) -> str:
    """同步包装器，供简单脚本调用"""
    import asyncio
    return asyncio.run(generate_animation_html(scenes))


# ── 便捷同步调用（供 test.py 使用） ─────────────────
def generate_script_sync(text: str) -> list[dict]:
    """同步包装器，方便在简单脚本中调用"""
    import asyncio
    return asyncio.run(generate_script(text))
