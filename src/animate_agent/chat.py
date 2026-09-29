"""Chatting with the model, and turning the conversation into something filmable.

Two jobs, one module. They are not two modules because neither has models,
validation or its own stage of the pipeline — the conversation lives in the
browser, and all this does is put a system prompt in front of it. `knowledge/` is
a package because it owns `models.py`, `validation.py` and `fidelity.py`; copying
that shape here would be copying the shape of the answer rather than the reason
for it.

**The two calls want opposite settings, and that is the whole reason `reply` and
`write_article` are separate functions rather than one with a flag.**

`reply` answers a person who is waiting, so it turns reasoning off — `llm.py`
records the measurement on `LLMConfig.thinking`: one call went from ~76s to
~5.5s. A chat box that goes quiet for a minute is not a chat box.

`write_article` is asked for a length, which is exactly the task that same
comment records as failing three times out of three with reasoning off. So it
keeps the default. One knob, two answers, and the difference is what the caller
is waiting for.

The article's numbers are not style advice; every one of them is arithmetic this
repository already does. See `_ARTICLE_RULES` for which.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from animate_agent.knowledge.prompts import narration_bounds, scene_count, word_budget
from animate_agent.llm import LLMClient, load_llm_config

_LOG = logging.getLogger(__name__)

#: How many turns of conversation may be sent at once, and how long each may be.
#: Enforced twice on purpose: `api.py` turns them into a declarative 422 at the
#: edge, and these are the numbers it uses, so the two cannot disagree.
MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 2000

#: Total characters across the conversation. The per-message cap bounds one turn;
#: this bounds the sum, which is what actually bounds the request.
MAX_TOTAL_CHARS = 12000

#: A conversational reply is a few sentences. The article is longer, and its
#: budget is the one thing the model must not spend on reasoning.
CHAT_MAX_TOKENS = 2048
ARTICLE_MAX_TOKENS = 8192

CHAT_TEMPERATURE = 0.7
ARTICLE_TEMPERATURE = 0.4

#: See the module docstring. `None` would mean "whatever the config says"; the
#: point of naming `"disabled"` here is that the fast path is what this call is
#: *for*, and a config change must not silently take it away.
CHAT_THINKING = "disabled"

#: How many times to ask for the article before giving up on it being the right
#: length. Two, matching the agents' habit of one corrective retry rather than a
#: loop: the failure is a model that ignored a number, and being told the number
#: again usually fixes it.
ARTICLE_ATTEMPTS = 2

#: The article is refused below this. It is not a style floor — a 60-character
#: "article" is a model that answered something else, and running the whole
#: three-minute chain on it produces a film about nothing.
MIN_ARTICLE_CHARS = 200

#: Above this, `write_article` warns rather than refuses. Refusing would throw
#: away a run over a number the downstream stage can still compress; the warning
#: is what tells the next person why the run after it took three attempts.
MAX_ARTICLE_CHARS = 650

#: How far under the knowledge layer's per-scene narration cap a section's body
#: should aim.
#:
#: Not a style preference — it is the difference between this endpoint working and
#: not. The knowledge layer rewrites each section into one scene's narration and
#: *refuses* anything over its cap, three times, and then fails the whole request.
#: An article written right at the cap leaves it no room to overshoot, and
#: overshooting by one character is an easy thing for a model to do: the first
#: real conversation this endpoint ever handled came back with a 61-character
#: narration against a 60-character cap, three attempts running. Five characters
#: of slack costs nothing downstream and is the whole margin.
ARTICLE_SCENE_HEADROOM = 5

CHAT_SYSTEM_PROMPT = """你是「智教」的助教，帮用户把「我想讲点什么」想清楚。

用户想要一条一两分钟的科普短片。你现在的任务不是写稿，是**问清楚**：
他到底想讲哪件事、讲给谁听、最想让人记住的是哪一句。

要求：
- 说人话。像坐在对面聊天，不是念稿子。
- 一次只问一两个问题，别一口气列八个。
- **短**。三四句话就够，用户不是来看你写作文的。
- 不确定的事实就问，或者直说自己不确定——不要编。
- 不要输出代码、表格、Markdown 标题。
- 用户说「生成动画」的时候，这条对话会被拿去写成稿子，所以你不用自己动手写。

如果用户一上来就把主题说得挺清楚，那就别硬问，帮他确认一下角度就行。"""


def article_body_bounds() -> tuple[int, int]:
    """How long one section's body should be, in characters.

    Exported because the prompt and the check that would refuse a run have to
    agree, and because the relationship is the thing worth holding: it is
    *strictly under* the knowledge layer's per-scene narration cap, with
    `ARTICLE_SCENE_HEADROOM` to spare. A test asserts the inequality rather than
    the numbers, so the band can move with the pipeline without the test having to
    be told twice.
    """
    _, cap = narration_bounds(script_mode=False)
    high = cap - ARTICLE_SCENE_HEADROOM
    return high - 15, high


def _article_rules() -> str:
    """The length arithmetic, computed rather than retyped.

    Every number in here is derived from the pipeline's own constants, the same
    way `knowledge.prompts._length_rules` derives its own. That matters more than
    it looks: this prompt is the *input* to the knowledge layer, and the knowledge
    layer refuses a document it cannot compress into its budget. If this prompt
    asks for a different length than the next stage allows, the failure surfaces
    three model calls later as a 422 that names nothing useful — after the money
    is spent.
    """
    budget = word_budget(90)
    scenes = scene_count(90)
    _, cap = narration_bounds(script_mode=False)
    low, high = article_body_bounds()
    return f"""- **第一行是标题，20 字以内，单独一行，前面不要加 `#` 或任何符号。**
- 正文 **{scenes} 节左右**，每节一个 `##` 小标题，下面**一段话**，不要多段。
- **每节正文 {low}~{high} 字。**
- **全文 {scenes * low}~{scenes * high} 字，绝对不要超过 {MAX_ARTICLE_CHARS} 字。**

这些数字是算术不是口味。下游把每一节改写成屏幕上的一幕，**一幕的旁白上限是 {cap} 字**，
所以每节要比那个上限短一截——写满了，下游改写时超出一个字就会被整个打回重来。
{budget} 字本来是 90 秒，{scenes} 节就是 {scenes} 幕。
写超了**不会让片子变长**——下游只会反复压缩，压不下去就整个失败。"""


ARTICLE_SYSTEM_PROMPT = f"""你是一位科普短片的撰稿人。

用户刚和你讨论了他想了解的内容。现在把这场讨论整理成**一条 90 秒科普片的文字稿**。

它接下来会被拆成一句一句的旁白、配上一屏一屏的画面，所以它必须**能画出画面**：
一条顺序、一个因果、一个前后对比、一个能演出来的过程。
通篇「很重要」「很关键」的稿子画不出任何东西，会被打回重做。

长度与结构：

{_article_rules()}

禁止出现：
- **标题行前面的 `#`。** 它的意思是「这里有一个章节」，而下游会因此多出一个没有内容的
  空章节，覆盖校验反复要求它被引用，然后整个请求失败。
- 代码块、图片、链接、表格。它们会被解析成讲不出来的内容块，然后卡住整条链路。
- `---` 分隔线。它会被当成文档头，把前面的内容整段吃掉。
- 罗列式要点清单。每个列表项都要被单独讲一遍，写三条不如写一段。
- 全角引号以外的装饰符号（`*`、`-` 开头的强调行）。正文就是普通段落。

写的时候：
- 中文、口语，像跟一个聪明的高中生说话。不要「首先/其次/综上所述」。
- 结论要带上它的前提（「忽略空气阻力时」这类条件不能省）。
- 只写这场讨论里出现过的事实，不补充没有依据的数字、名称或结论。
- 不要写「本文」「下面我们来看看」——没有人会念它们。

直接输出 Markdown 正文。不要解释，不要用 ``` 包起来。"""


def history(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """The conversation as the chat-completions API wants it.

    Trims to the last `MAX_MESSAGES` turns: a long conversation's *end* is what
    the next reply depends on, so dropping from the front loses the least. Each
    message is truncated to `MAX_MESSAGE_CHARS` for the same reason one message at
    a time is capped at the edge — a single pasted essay should not be able to
    push the whole request over the model's context.
    """
    kept: list[dict[str, str]] = []
    for message in list(messages)[-MAX_MESSAGES:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()[:MAX_MESSAGE_CHARS]
        if content:
            kept.append({"role": role, "content": content})
    return kept


async def _ask(
    system: str,
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    label: str,
    thinking: str | None,
    client: LLMClient | None = None,
) -> str:
    """One completion with `system` in front. Owns its client when not given one.

    Building and closing a client per call is what `knowledge/service.py` does,
    and it is right here for the same reason: an `httpx.AsyncClient` held open
    between requests is a connection pool nobody closes. The conversation being
    long-lived does not make the *client* long-lived — the transcript is resent
    every turn, which is the same deal every stateless chat backend makes.
    """
    owned = client is None
    llm = client or LLMClient(load_llm_config())
    try:
        return await llm.chat(
            [{"role": "system", "content": system}, *history(messages)],
            temperature=temperature,
            max_tokens=max_tokens,
            label=label,
            thinking=thinking,
        )
    finally:
        if owned:
            await llm.aclose()


async def reply(messages: Sequence[Mapping[str, str]], *, client: LLMClient | None = None) -> str:
    """One conversational turn. Fast on purpose — see the module docstring."""
    return await _ask(
        CHAT_SYSTEM_PROMPT,
        messages,
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS,
        label="对话",
        thinking=CHAT_THINKING,
        client=client,
    )


def article_problem(text: str) -> str:
    """What is wrong with `text` as an article, or `""` if nothing is.

    Only the two ends are checked, and only the short end is fatal. A model that
    wrote 60 characters answered a different question, and running the pipeline
    on it produces a film about nothing at real cost. A model that wrote 900
    characters overshot a number it was told once; the downstream stage still
    compresses documents for a living, so the honest move is to say so and let it
    try rather than to throw the run away.
    """
    body = text.strip()
    if len(body) < MIN_ARTICLE_CHARS:
        return f"太短了：只有 {len(body)} 字，至少要 {MIN_ARTICLE_CHARS} 字"
    if len(body) > MAX_ARTICLE_CHARS:
        return f"太长了：{len(body)} 字，上限是 {MAX_ARTICLE_CHARS} 字"
    return ""


async def write_article(
    messages: Sequence[Mapping[str, str]], *, client: LLMClient | None = None
) -> str:
    """Turn the conversation into the Markdown source the pipeline ingests.

    Retries **once** when the length is wrong, by saying which way it was wrong —
    the same shape as the agents' retry loop, and for the same reason: the model
    was given the number in prose and stopped holding it, so the fix is to give it
    the number again next to what it actually produced.

    Reasoning stays on. `LLMConfig.thinking` records that disabled reasoning
    failed a length-constrained task three times out of three, and a length
    constraint is precisely what this call has.
    """
    history_messages = history(messages)
    problem = ""
    text = ""
    for attempt in range(1, ARTICLE_ATTEMPTS + 1):
        asking: list[dict[str, str]] = [
            {"role": "user", "content": "请把上面的讨论整理成这条科普片的文字稿。"}
        ]
        if problem:
            asking.append(
                {
                    "role": "user",
                    "content": (
                        f"上一次的稿子{problem}。请重写一遍，"
                        f"严格按前面说的字数和节数来。只输出 Markdown 正文。"
                    ),
                }
            )
        text = await _ask(
            ARTICLE_SYSTEM_PROMPT,
            [*history_messages, *asking],
            temperature=ARTICLE_TEMPERATURE,
            max_tokens=ARTICLE_MAX_TOKENS,
            label=f"写稿(第{attempt}次)",
            thinking=None,
            client=client,
        )
        problem = article_problem(text)
        if not problem:
            return text.strip()
        _LOG.warning("写稿第 %d 次不合格：%s", attempt, problem)

    if len(text.strip()) < MIN_ARTICLE_CHARS:
        raise ValueError(f"模型写不出可用的稿子：{problem}")
    _LOG.warning("写稿长度仍不合格（%s），照跑。下游可能要重试几轮。", problem)
    return text.strip()
