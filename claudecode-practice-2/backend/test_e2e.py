"""端到端测试 — 模拟完整流程，定位哪一步失败"""
import asyncio, json, sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import extract_text
from script_generator import generate_script, generate_animation_html

TEST_FILE = None
# 自动找一个测试文件
for ext in [".pptx", ".docx", ".pdf"]:
    candidate = os.path.join(os.path.dirname(__file__), "test_sample" + ext)
    if os.path.exists(candidate):
        TEST_FILE = candidate
        break

if not TEST_FILE:
    print("❌ 未找到测试文件，请在 backend/ 下放一个 test_sample.pptx/.docx/.pdf")
    sys.exit(1)

async def main():
    total_start = time.time()
    print(f"📂 测试文件: {TEST_FILE}")
    print("=" * 50)

    # Step 1: 提取文本
    step_start = time.time()
    try:
        text = extract_text(TEST_FILE)
        t = time.time() - step_start
        print(f"✅ [Step 1] 文本提取 — {len(text)} 字符 ({t:.1f}s)")
        print(f"   前100字: {text[:100]}")
    except Exception as e:
        print(f"❌ [Step 1] 文本提取失败 ({time.time()-step_start:.1f}s): {e}")
        import traceback; traceback.print_exc()
        return

    # Step 2: 生成场景
    step_start = time.time()
    try:
        scenes = (await generate_script(text)).model_dump()["scenes"]
        t = time.time() - step_start
        print(f"✅ [Step 2] 场景生成 — {len(scenes)} 个场景 ({t:.1f}s)")
        for s in scenes:
            print(f"   Scene {s['scene']}: {s['title']}")
    except Exception as e:
        print(f"❌ [Step 2] 场景生成失败 ({time.time()-step_start:.1f}s): {e}")
        import traceback; traceback.print_exc()
        return

    # Step 3: 生成动画
    step_start = time.time()
    try:
        html = await generate_animation_html(scenes)
        t = time.time() - step_start
        if html and len(html) > 500:
            out = os.path.join(os.path.dirname(__file__), "..", "frontend", "generated_animation.html")
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ [Step 3] 动画生成 — {len(html)} 字符 ({t:.1f}s)")
        else:
            print(f"❌ [Step 3] HTML 太短: {len(html)} 字符")
    except Exception as e:
        print(f"❌ [Step 3] 动画生成失败 ({time.time()-step_start:.1f}s): {e}")
        import traceback; traceback.print_exc()
        return

    total_t = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"🎉 全部完成! 总耗时: {total_t:.1f}s")

asyncio.run(main())
