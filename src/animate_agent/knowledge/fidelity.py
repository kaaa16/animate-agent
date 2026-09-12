"""Optional second-pass fidelity check for a generated LessonIR.

`_uncovered_ids` in the agent proves every source block was *referenced*. It
cannot prove the block was actually *taught*: a scene can cite a parameters
block and still drop three of its definitions. Catching that needs a semantic
comparison, which is what this module prompts for.

The check costs a second LLM call per generation, so it is off by default
(`knowledge.verify_fidelity`). Its prompt is deliberately biased toward
under-reporting — a verifier that flags normal summarising would trigger
endless retries.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.prompts import build_knowledge_prompt

FIDELITY_SYSTEM_PROMPT = """你是一位严格的事实核对员。用户会给你两份材料：
一份是原始文档（结构化形式），另一份是据它生成的课程 JSON。
你的任务是找出课程 JSON 相对于原始文档的两类问题，输出为一个 JSON 对象
（不要输出 JSON 以外的任何文字）：

{
  "dropped_facts": ["原文有、但课程里没说或说漏的内容"],
  "added_claims": ["课程里出现、但原文没有的事实、数字、结论、命名或修辞"]
}

判定标准：
- dropped_facts：原文给出的定义、结论、限定条件（例如「在忽略空气阻力的前提下」）、
  具体条目，如果在课程里被省略，或者只写了个标签而没讲内容，就算漏。
  但课程本来就是提炼：正常的概括、合并、顺序调整不算漏，
  只报「原文明确说了、课程却完全没说」的实质内容。
- added_claims：课程里出现而原文没有的事实、数字、结论、命名或修辞比喻。
  为了把逻辑讲通而做的连接性说明（例如点明某一步骤的目的）不算新增；
  但如果它引入了原文没有的具体事实，或者「全部规律」这类绝对化结论，就要报。
- 没有问题就输出空数组。宁可少报，也不要为了凑数把正常概括报成问题。

只输出 JSON 对象，不要用 ```json 包裹，不要输出任何解释性文字。"""


class FidelityReport(BaseModel):
    """What the verifier found wrong with a candidate LessonIR."""

    model_config = ConfigDict(extra="forbid")

    dropped_facts: list[str] = Field(default_factory=list)
    added_claims: list[str] = Field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.dropped_facts or self.added_claims)

    def describe(self) -> str:
        """Render the findings as retry feedback for the generating agent."""
        parts: list[str] = []
        if self.dropped_facts:
            parts.append("遗漏了原文内容：" + "；".join(self.dropped_facts))
        if self.added_claims:
            parts.append("出现了原文没有的内容：" + "；".join(self.added_claims))
        return "；".join(parts)


def _lesson_payload(lesson: LessonIR) -> dict[str, Any]:
    return {
        "title": lesson.title,
        "summary": lesson.summary,
        "learning_objectives": lesson.learning_objectives,
        "scenes": [
            {
                "id": scene.id,
                "title": scene.title,
                "objective": scene.objective,
                "narration": scene.narration,
                "key_points": scene.key_points,
            }
            for scene in lesson.scenes
        ],
    }


def build_fidelity_prompt(document: DocumentIR, lesson: LessonIR) -> str:
    """Lay the source and the generated course side by side for the verifier."""
    return "\n".join(
        [
            "# 原始文档",
            "",
            build_knowledge_prompt(document),
            "# 课程 JSON",
            "",
            json.dumps(_lesson_payload(lesson), ensure_ascii=False, indent=2),
        ]
    )
