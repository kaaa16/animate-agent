"""
IR 中间表示层（Intermediate Representation）。

定义教学动画生成管线的三段数据契约，对应目标架构：

    URL/File → Ingest → DocumentIR → Knowledge Agent → LessonIR
             → Storyboard Agent → StoryboardIR → { SVG / Three.js / Manim }

设计原则：
- StoryboardIR 是「渲染器无关」的：只描述语义对象与步骤状态（role/props），
  具体画法由各渲染器自行映射（场景模板 scene_type → 渲染器内的 builder）。
- renderer_hint 仅作推荐，可被上层（用户 / API 参数）覆盖，便于以后切换手动选择。
- 字段全部给默认值、默认忽略多余字段，保证 AI 输出有出入时不至于解析失败。
"""

from pydantic import BaseModel
from typing import List, Dict, Any


# ── 阶段1：DocumentIR（Ingest 输出）─────────────────────────

class Section(BaseModel):
    """文档中的一个章节 / 分节。"""
    heading: str = ""                 # 章节标题（pptx 标题 / docx heading / pdf 页标题）
    paragraphs: List[str] = []        # 该章节下的段落文本


class DocumentIR(BaseModel):
    """Ingest 阶段输出：结构化文档。"""
    source_type: str = ""             # "pptx" | "docx" | "pdf" | "url" | "txt"
    title: str = ""                   # 课件 / 文档标题
    sections: List[Section] = []      # 结构化章节
    raw_text: str = ""                # 全文纯文本（供下游兜底）


# ── 阶段2：LessonIR（Knowledge Agent 输出）──────────────────

class LessonScene(BaseModel):
    """一个教学场景（课程设计粒度）。"""
    scene: int = 0                    # 序号，从 1 开始
    title: str = ""                   # 场景标题
    narration: str = ""               # 讲解文案
    key_points: List[str] = []        # 关键词（约 3 个）


class LessonIR(BaseModel):
    """Knowledge Agent 输出：教学场景脚本。"""
    scenes: List[LessonScene] = []


# ── 阶段3：StoryboardIR（Storyboard Agent 输出，渲染器无关）──

class StoryboardObject(BaseModel):
    """一个语义对象。渲染器根据 role 映射到各自图元。"""
    id: str = ""                      # 唯一 ID，如 "ball"、"arrow_vx"
    role: str = ""                    # 语义角色：projectile / ground / force_arrow / ...
    position: List[float] = []        # [x, y, z] 或 [x, y]
    props: Dict[str, Any] = {}        # 角色相关参数：initialSpeed / angle / color / size / ...


class StoryboardStep(BaseModel):
    """一个动画步骤（分阶段演示）。"""
    id: str = ""                      # 唯一 ID
    title: str = ""                   # 步骤标题
    description: str = ""             # 讲解文案
    highlights: List[str] = []        # 需要高亮的对象 ID
    object_states: Dict[str, Any] = {}  # {object_id: {状态字段}}


class StoryboardScene(BaseModel):
    """一个场景的完整故事板。"""
    scene_type: str = "generic"       # 语义场景模板：projectile / force / lidar / wave / circuit / ...
    renderer_hint: str = "threejs"    # 推荐渲染器：threejs | svg | manim（可被覆盖）
    objects: List[StoryboardObject] = []
    steps: List[StoryboardStep] = []


class StoryboardIR(BaseModel):
    """Storyboard Agent 输出：供渲染器消费的完整故事板。"""
    title: str = ""                   # 顶层标题（对应原 meta.title）
    subject: str = ""                 # 学科（对应原 meta.subject）
    eyebrow: str = ""                 # 顶部小字（对应原 meta.eyebrow）
    scenes: List[StoryboardScene] = []
