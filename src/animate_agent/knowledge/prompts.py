"""Prompt templates for the Knowledge Agent."""

from __future__ import annotations

from animate_agent.documents.models import DocumentIR
from animate_agent.rendering.layout import SPEECH_CHARS_PER_SECOND

#: How many characters a scene's narration aims for, which is what turns a word
#: budget into a scene count.
#:
#: 70 is four beats of one screen each: a beat holds 2.8 seconds at six
#: characters a second, and one screen holds about 17 characters — one line at
#: `caption.js`'s 20px in a 920px band, read at a glance rather than squinted at.
#:
#: This is the number that was missing when the captions came out as labels. The
#: word budget was fixed at 350 characters and the scene count was left to the
#: model's judgement, so a 350-character script kept being cut into nine scenes
#: and thirty-three captions of ten characters apiece: every one of them a
#: fragment, none of them a sentence, and the film's length never changed no
#: matter what was done to the writing.
SCENE_CHARACTERS = 70

#: The film's length when a caller did not say what it is.
#:
#: Used for the *prompt* only, never as a check: an agent built by hand — a test,
#: a tool — still gets told what it is writing towards, and is not refused for
#: producing a twenty-second lesson, which is what every fixture in the suite is.
#: `config.StoryboardSettings.target_seconds` is the number a real run uses, and
#: `service.generate_lesson` is what passes it.
DEFAULT_FILM_SECONDS = 90.0


def narration_bounds(script_mode: bool) -> tuple[int, int]:
    """How long one scene's narration may be, in characters, per mode.

    **Two bands because the two jobs are different, and the schema cannot say
    so.** A document found in the world gets compressed into the film's word
    budget, and 20~60 is the band that has done that since the budget existed: a
    scene past 60 characters is a scene that has not decided what it is about.

    A script written *for* this film is not compressed at all — the film is as
    long as the script — so its scenes are as long as the author made them, and
    110 is where a scene stops being a scene rather than a measurement of
    anything. Twelve of those is about four minutes, which is what `max_scenes`
    and the film's ceiling are there to bound.

    Exported because the prompt and the check that refuses a run have to agree.
    They did not: the prompt said 9~14 characters while the schema said 8~40, the
    model was told both, and nothing read either one back. A band the prompt
    states and the validator does not enforce is a band that drifts.
    """
    return (40, 110) if script_mode else (20, 60)


def word_budget(target_seconds: float) -> int:
    """How many characters the whole script has room for.

    The one place the film's length turns into words, and it is a division:
    `beat_duration` holds a beat for `len / SPEECH_CHARS_PER_SECOND` seconds, so a
    film of `target_seconds` of speech is `target_seconds × 6` characters. Every
    other number this layer works with — how many scenes, how long each one — is
    derived from this one, which is why it is computed and written into the prompt
    rather than being a constant kept in step with the film's target by hand. It
    *was* a constant, 350, and it had already drifted: the lesson on disk sums to
    368 characters while the prompt told the model 350 was the ceiling.
    """
    return round(target_seconds * SPEECH_CHARS_PER_SECOND)


def scene_count(target_seconds: float) -> int:
    """How many scenes the word budget buys, at `SCENE_CHARACTERS` apiece."""
    return max(round(word_budget(target_seconds) / SCENE_CHARACTERS), 1)


_KNOWLEDGE_HEAD = """你是一位资深课程设计师。
用户会给你一份已经解析成结构化形式的文档（包含标题，以及若干 section，
每个 section 有 id 和若干内容块）。

你的任务是理解这份文档，并把它重新组织成一份面向初学者的教学课程大纲，
输出为一个 JSON 对象。JSON 必须包含以下字段（不要输出 JSON 以外的任何文字）：

{
  "title": "课程标题",
  "subject": "学科/主题",
  "summary": "全文一句话摘要",
  "learning_objectives": ["整体学习目标1", "整体学习目标2", "..."],
  "scenes": [
    {
      "title": "场景标题",
      "objective": "本节教学目标（一句话）",
      "narration": "讲解文案（口语化、面向初学者）",
      "key_points": ["关键词1", "关键词2", "..."],
      "source_refs": ["section-xxx", "section-xxx-block-1", "..."]
    }
  ]
}

要求：
- learning_objectives 写 2~4 条。
"""

_KNOWLEDGE_TAIL = """- 每个场景必须覆盖一个「完整的语义单元」——一个概念、一个步骤或一条注意事项。
  若某个候选场景只能覆盖零散的半句内容，请与相邻场景合并，
  不要把一步拆成多个内容很薄的场景。
- **多讲「为什么」，少报「是什么」。** 每一个概念按这个顺序讲：
  **没有它会怎样 → 它是怎么工作的 → 所以你能得到什么。**
  观众不需要你告诉他画面上已经画着的东西；他需要的是那个画面上画不出来的理由。
  「JSON 的扩展名是 .json」是白说，「扩展名是给人和工具一眼认出来的」才值得占一屏。
- **口语，允许有观点。** 像一个觉得这件事有意思的人，在跟一个聪明的朋友说话。
  可以用「这里最要紧」「听着像废话，其实不是」这样的判断句；
  不要用「首先/其次/综上所述」这类书面连接词，不要有括号里的补充说明，
  不要预告你接下来要讲什么（「下面我们来看……」）。
- **句子长短要有节奏。** 连续三句话一样长、一样的形状，观众会觉得困，
  哪怕内容是对的。一句长的后面跟一句短的。
- key_points 写 2~12 个关键词，要把该场景讲到的要点全部列进去；
  原文列了几项就列几项（上限 12 项），不要漏。
- source_refs 必须引用输入文档中真实存在的 section id 或 block id，用于溯源；
  请列出该场景讲解所依据的「全部」block id，不要只列其中一部分；确实没有对应来源才写空数组。
- 覆盖完整性：输入文档中「每一个」block 的实质内容，都必须至少被一个场景的
  source_refs 引用到，不允许遗漏（例如原文列了 5 件工具，场景里就要写全 5 件，
  不能少一件）。section 只要它下面的 block 都被引用了就算覆盖，不必再单独引用
  section id。漏掉会被校验打回重做。
- 忠于原文：只能使用输入文档中出现的事实。允许为了把逻辑讲通做「连接性说明」
  （例如点明某个步骤的目的、说明两个量为什么相关），但不得引入原文没有的事实、
  数字、结论或命名；也不要用「全部规律」这类绝对化的说法。
  **比方另算**：原文自己用了某个比方（「内存像书架」），就接着把它用下去，
  用到底、别中途换一个；原文没有的比方不要自己造——造出来的比方是一个关于世界的断言，
  它可能不准，而这条片子的底线是准确。
- 限定条件必须保留：「在忽略空气阻力的前提下」这类前提是结论成立的条件，
  不能省略，也不能改写成别的说法。
- 原文给出的定义和结论要转述完整，不能只留一个标签——
  原文写了「与物体质量无关」，就要把这句结论说出来，而不是只写个小标题了事。
- 同一个场景内，objective、key_points、narration 必须自洽：
  objective 写「六个关键参数」，key_points 就要列满六个，narration 也要讲全。
- 若原文本身存在看似矛盾之处（例如既说"不用粘合剂"又说"可涂木工胶"），
  请在 narration 中把两者的关系讲清楚，不要制造新的矛盾。
- 直接输出 JSON 对象，不要用 ```json 包裹，不要输出任何解释性文字。"""


def _length_rules(*, script_mode: bool, target_seconds: float) -> str:
    """The part of the prompt that is arithmetic rather than instruction.

    A function because this is the one block that cannot be a constant: it quotes
    the film's target, the characters that target buys, and the scene count those
    characters buy, and all three move together. While it was prose inside a
    constant it said one minute and 350 characters; the target moved to ninety
    seconds and the sentence did not follow, which is a drift nothing catches —
    no test reads a prompt, so a stale number in one is a number that lies to the
    model forever.

    The source-mode paragraph is here rather than in the tail for the same
    reason: whether the text may be rewritten is what the length rule *means*,
    and the two sentences read as one thought.
    """
    budget = word_budget(target_seconds)
    scenes = scene_count(target_seconds)
    low, high = narration_bounds(script_mode)
    if script_mode:
        source = (
            "- **这份稿子是为了做这条片子写的，不是随手拿到的资料。** 照稿子的原话和顺序走："
            "不要改写措辞，不要合并段落，不要为了凑长度往里加话。"
            "稿子长片子就长，稿子短片子就短——长度是写稿的人定的，你只负责把它排成几幕。"
        )
    else:
        source = (
            "- **这份文档不是你写的，你可以重组它。** 顺序可以调、句子可以改写、"
            "和主题无关的部分可以不讲——但目标是上面那个字数，不是把文档抄一遍。"
        )
    return f"""- **先说长度，因为后面每一个决定都是从它推出来的。**
  这是一条 **{target_seconds:.0f} 秒**左右的科普片（大约一到两分钟）。画面上的字幕按
  **每秒 6 个字**推进，换幕还要再占掉几秒，所以全片 narration 一共就是 **{budget} 字**上下。
- **幕数由字数推出来，不要凭感觉定。** 一幕大约 **{SCENE_CHARACTERS} 字**，也就是四拍、
  每拍一屏——一屏 17 字上下，一眼就读完了。所以 {budget} 字就是 **{scenes} 幕**左右。
  不要因为「内容多」就多分几幕：**多一幕不占时间，字多才占时间**。
- **每幕的 narration 写 {low}~{high} 字**，只讲这一幕最核心的一件事。宁可少分几幕、
  每幕把这件事讲透，也不要一幕里只丢下一个标签。
{source}
"""


def build_knowledge_system_prompt(
    *, script_mode: bool = False, target_seconds: float = DEFAULT_FILM_SECONDS
) -> str:
    """The system prompt for one run: the rules, plus this run's length budget."""
    return (
        _KNOWLEDGE_HEAD
        + _length_rules(script_mode=script_mode, target_seconds=target_seconds)
        + _KNOWLEDGE_TAIL
    )


def build_knowledge_prompt(document: DocumentIR) -> str:
    """Serialize a DocumentIR into compact structured text for the LLM."""
    lines: list[str] = [f"标题：{document.title}", ""]
    for section in document.sections:
        lines.append(f"## [{section.id}] {section.title}")
        for block in section.blocks:
            lines.append(f"  [{block.id}] {block.text}")
        lines.append("")
    return "\n".join(lines)
