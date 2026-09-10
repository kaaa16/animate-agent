"""测试新 prompt — 单场景、短动画，验证概念具象化效果"""
import asyncio, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_generator import generate_animation_html

# 两个不同风格的测试场景，验证模型是否真的"用画面讲故事"
scenes = [
    {
        "scene": 1,
        "title": "雷达避障",
        "narration": "智能小车在行驶过程中通过车顶的雷达发射信号，探测前方障碍物。当雷达波遇到石头反射回来，小车自动计算出障碍物的位置并转向避开。",
        "key_points": ["雷达发射", "信号反射", "自动避障"]
    },
]

async def main():
    print("=" * 60)
    print("测试新 prompt — 概念具象化动画")
    print("场景: 雷达避障（期望看到小车+雷达波+障碍物+转向动画）")
    print("=" * 60)

    start = time.time()
    try:
        html = await generate_animation_html(scenes)
        t = time.time() - start

        if html and len(html) > 500:
            out = os.path.join(os.path.dirname(__file__), "..", "frontend", "generated_animation.html")
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n✅ 成功! {len(html)} 字符, 耗时 {t:.1f}s")
            print(f"✅ 已保存到: frontend/generated_animation.html")
            print(f"\n--- HTML 前 600 字符预览 ---")
            print(html[:600])
        else:
            print(f"\n❌ HTML 太短: {len(html)} 字符")
            print(html[:2000])
    except Exception as e:
        print(f"\n❌ 失败 ({time.time()-start:.1f}s): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
