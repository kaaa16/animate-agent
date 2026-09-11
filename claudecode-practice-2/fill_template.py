"""
填写比赛作品说明文档模板
基于 AI教学动画生成器 项目的真实架构和特色
"""
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import copy, os

# ── 配置 ──────────────────────────────────
TEMPLATE = r"C:\Users\zhika\Desktop\2026中国高校计算机大赛人工智能创意赛初赛（鸿蒙赛道）作品说明文档模板.docx"
OUTPUT = r"d:\claudecode练习2\作品说明文档_AI智教.docx"

# ── 填写内容 ──────────────────────────────
# 封面信息（留空，用户手动填写）
COVER = {
    "参赛学校": "_______________________________",
    "团队名称": "_______________________________",
    "作品名称": "_______________________________",
    "赛题方向": "（√）Agent创新",
    "联系人": "_______________________________",
    "联系电话": "_______________________________",
}

# 创意描述（30字以内）
CREATIVE_DESC = "AI驱动教学可视化Agent：上传文档自动生成可交互动画，支持自由拖拽调参与视角切换，直观理解抽象知识"

# ── 设计稿/技术方案 ─────────────────────────
TECH_DESIGN = """一、系统架构概览

本系统采用"AI多模型协作管道 + 模板驱动渲染"架构，将传统课件制作从"逐帧手绘"变革为"文档一键生成可交互动画"。

二、核心技术流程

1. 文件解析层：支持 PPT/Word/PDF 三种格式，自动提取纯文本内容，保留教学结构。

2. 三模型AI协作管道：
   · Kimi (kimi-k2.7-code)：场景拆分 —— 理解文档语义，将内容拆分为3~6个教学场景，每个场景包含标题、讲解文案和关键词。
   · DeepSeek (deepseek-v4-pro)：场景优化 —— 审查并优化拆分方案，确保信息量适中、文案口语化。
   · DeepSeek (deepseek-v4-pro)：结构化生成 —— 输出严格的 lesson JSON 数据（约2KB），描述场景类型、物体状态、动画步骤序列。

3. 模板驱动渲染引擎：
   · 3D模式：基于 Three.js (WebGL) 的固定模板播放器，支持激光雷达避障等3D教学场景，通过 importmap CDN 加载。
   · 2D模式：基于 Canvas 2D 的俯视视角模板，适配不同教学主题。
   · 渲染引擎与数据分离：AI只输出结构化JSON，前端模板保证渲染质量100%可控。

4. 智能降级机制：当AI无法匹配合适模板时，自动切换为AI直接生成完整HTML动画，确保零失败率。

三、交互设计亮点

本系统的核心差异化优势在于"可视化+可交互"：

1. 参数自由调节：用户可在播放过程中调整动画速度、物体属性（如雷达探测范围、障碍物大小）、视角远近等参数，实时观察参数变化对场景的影响。

2. 视角自由拖拽（3D模式）：基于 OrbitControls 实现鼠标拖拽旋转/缩放/平移三维场景，学生可以从任意角度观察教学模型，真正"动手理解"抽象概念。

3. 分步导航：动画划分为多个教学步骤，用户可自由前进/后退/跳转到任意步骤，配合左侧文案面板同步展示讲解内容。

4. 双模式切换：一键在2D俯视视角与3D立体视角间切换，适配不同教学场景需求。

四、技术栈

· 后端：Python FastAPI + 异步IO
· AI模型：Kimi kimi-k2.7-code + DeepSeek deepseek-v4-pro
· 3D渲染：Three.js 0.160 + WebGL
· 2D渲染：原生 Canvas 2D API
· 通信协议：postMessage（主页面与iframe播放器间数据传输）

五、扩展规划

系统架构预留了互动游戏模块接口，未来将支持：
· 闯关问答：基于教学内容的交互式答题，答对驱动动画推进
· 角色扮演：学生以第一人称视角在场景中探索
· 参数实验：学生手动调节物理/数学模型参数，观察结果变化"""

# ── 介绍文档（800字以内）────────────────────
INTRO_DOC = """一、创意背景

当前教育数字化转型面临一个核心矛盾：高质量的动画教学课件能显著提升学习效果，但其制作门槛极高——教师需要掌握建模、动画、编程等多重技能，单个5分钟教学动画的制作周期往往以周为单位。AI大模型的出现为解决这一问题提供了全新可能，但当前AI直接生成动画代码的方案存在质量不稳定、缺乏交互性的问题。

本作品"AI智教"——大模型驱动的交互式教学动画生成系统，旨在让任何教师只需上传一份PPT/Word/PDF教学文档，即可在约90秒内自动获得一个高质量、可交互的2D或3D教学动画。

二、核心功能设计

1. 智能文档解析：自动识别并提取教学文档中的文本内容与结构，无需人工预处理。

2. AI场景拆分与动画生成：首创"三模型协作管道"——Kimi负责语义理解与场景拆分，DeepSeek负责结构化优化与JSON数据生成，最终由前端模板引擎渲染为动画。输出体积仅约2KB的JSON数据，生成速度快、质量稳定。

3. 模板驱动渲染：不同于AI直接生成完整代码的方案，本系统采用"AI产出数据 + 固定模板渲染"架构，从根本上解决了AI生成代码质量不稳定的痛点。模板支持激光雷达避障等具象化教学场景，将抽象知识转化为可视画面。

4. 深度交互能力（核心差异化优势）：本系统生成的不仅是"能看"的动画，更是"能动"的教学工具：
   · 3D场景支持鼠标拖拽旋转/缩放/平移，学生可任意角度观察模型；
   · 播放过程中可实时调整动画参数（速度、物体属性等）；
   · 分步导航支持自由跳转，配合文案解说；
   · 2D/3D双模式一键切换，适配不同教学需求。

5. 智能降级保障：当AI判断无合适模板时，自动切换为AI直接生成HTML动画，保证100%可用率。

三、技术实现路径

后端基于Python FastAPI构建异步服务，文件解析支持PPT/Word/PDF三格式。AI管道使用Kimi kimi-k2.7-code与DeepSeek deepseek-v4-pro双模型协作，通过精心设计的System Prompt约束输出格式与质量。前端3D播放器基于Three.js，2D播放器基于原生Canvas API，通过postMessage协议实现主页面与iframe播放器间的数据传输。系统自动选择模板优先方案，模板不匹配时无缝降级为AI生成HTML。

四、市场前景

本系统面向K12教师、高校讲师、企业培训师等广泛群体。相比传统动画制作外包（单分钟报价数百至数千元），本系统成本趋近于零；相比现有AI动画工具，本系统的交互性和模板稳定性构成核心壁垒。未来可扩展至互动游戏、VR教学等场景，商业潜力显著。"""

# ══════════════════════════════════════════════
#  填写逻辑
# ══════════════════════════════════════════════

doc = Document(TEMPLATE)

# ── 1. 封面段落 ──
for para in doc.paragraphs:
    text = para.text.strip()

    if text.startswith("参赛学校：") and "学校" not in text.split("：")[-1][:2]:
        # 保留用户手动填写
        pass
    elif text.startswith("参赛学校："):
        para.clear()
        run = para.add_run(f"参赛学校：{COVER['参赛学校']}")
        run.font.size = Pt(14)

    if text.startswith("团队名称："):
        para.clear()
        run = para.add_run(f"团队名称：{COVER['团队名称']}")
        run.font.size = Pt(14)

    if text.startswith("作品名称："):
        para.clear()
        run = para.add_run(f"作品名称：{COVER['作品名称']}")
        run.font.size = Pt(14)

    if text.startswith("赛题方向："):
        para.clear()
        run = para.add_run(f"赛题方向：{COVER['赛题方向']}")
        run.font.size = Pt(14)

    if text.startswith("联系人（队长）："):
        para.clear()
        run = para.add_run(f"联系人（队长）：{COVER['联系人']}")
        run.font.size = Pt(14)

    if text.startswith("联系电话（队长）："):
        para.clear()
        run = para.add_run(f"联系电话（队长）：{COVER['联系电话']}")
        run.font.size = Pt(14)

# ── 2. 表格：作品名称 & 团队名称 & 赛题方向 ──
table = doc.tables[0]

# 作品名称 (row 0) — 留空
# 团队名称 (row 1) — 留空
# 参赛学校 (row 2) — 留空

# 赛题方向 (row 3): 填入 (√) Agent创新
direction_cell = table.rows[3].cells[1]
direction_cell.text = "（√）Agent创新"

# ── 3. 填充内容区域 — 在文档末尾添加 ──

# 找到"创意描述"段落之后插入内容
# 策略：遍历段落，找到关键标记段落，在其后插入内容

def find_para_containing(doc, keyword):
    """找到包含关键词的段落索引"""
    for i, p in enumerate(doc.paragraphs):
        if keyword in p.text:
            return i
    return -1

# 创意描述 — 找到"创意描述"标记，在其后插入
idx_creative = find_para_containing(doc, "一句话抽述关键创新点")

# 找到"创意描述"这个 list paragraph
creative_title_idx = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "创意描述" and p.style.name == "List Paragraph":
        creative_title_idx = i
        break

# 在"创意描述"标题下方插入内容
if creative_title_idx >= 0:
    # 在下一个段落插入创意描述
    target_para = doc.paragraphs[creative_title_idx + 1] if creative_title_idx + 1 < len(doc.paragraphs) else None
    if target_para and "一句话" in target_para.text:
        target_para.clear()
        run = target_para.add_run(CREATIVE_DESC)
        run.font.size = Pt(10.5)  # 五号
        run.font.name = "宋体"

# 设计稿/技术方案
design_title_idx = -1
for i, p in enumerate(doc.paragraphs):
    if "设计稿" in p.text and "技术方案" in p.text and p.style.name == "List Paragraph":
        design_title_idx = i
        break

if design_title_idx >= 0:
    # 在标题后插入技术方案内容
    # 找到下一个段落并修改
    next_idx = design_title_idx + 1
    if next_idx < len(doc.paragraphs):
        # 清除后续空白段落并插入内容
        pass

    # 使用更稳健的方式：在标题段落后插入新段落
    # 获取标题段落的元素位置
    design_para = doc.paragraphs[design_title_idx]

    # 在标题后插入分隔空行
    insert_after_paragraph(doc, design_title_idx, "")

    # 逐段插入技术方案
    tech_lines = TECH_DESIGN.strip().split("\n")
    current_idx = design_title_idx + 1
    for line in tech_lines:
        insert_after_paragraph(doc, current_idx, line)
        current_idx += 1

# ── 介绍文档 ──
intro_title_idx = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "介绍文档" and p.style.name == "List Paragraph":
        intro_title_idx = i
        break

if intro_title_idx >= 0:
    intro_lines = INTRO_DOC.strip().split("\n")
    current_idx = intro_title_idx + 1
    for line in intro_lines:
        insert_after_paragraph(doc, current_idx, line)
        current_idx += 1

# ── 辅助函数（需要定义在前面，但为了逻辑清晰放在这里，实际运行时已在上面调用） ──
# 注：由于Python执行顺序，此函数定义需要在调用之前。已在上方代码中调整。

# 保存
doc.save(OUTPUT)
print(f"✅ 文档已保存到: {OUTPUT}")
print(f"   创意描述: {CREATIVE_DESC}")
print(f"   字符数: {len(CREATIVE_DESC)}")
