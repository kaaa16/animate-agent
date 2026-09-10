"""测试 DeepSeek 动画生成 — 单独测试，快速定位问题"""
import asyncio, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_generator import generate_animation_html

scenes = [
    {"scene": 1, "title": "什么是AI", "narration": "人工智能就像给计算机装上了大脑，让它能够像人类一样思考和学习。", "key_points": ["人工智能", "机器学习", "深度学习"]},
    {"scene": 2, "title": "AI的应用", "narration": "从手机语音助手到自动驾驶汽车，AI技术已经深入我们生活的方方面面。", "key_points": ["语音助手", "自动驾驶", "智能推荐"]},
]

async def main():
    print("=" * 50)
    print("开始测试 DeepSeek 动画生成...")
    print("=" * 50)
    try:
        html = await generate_animation_html(scenes)
        if html and len(html) > 500:
            # 保存
            out = os.path.join(os.path.dirname(__file__), "..", "frontend", "generated_animation.html")
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n✅ 成功! HTML 长度: {len(html)} 字符")
            print(f"✅ 已保存到: {out}")
            print(f"\n前200字符预览:\n{html[:200]}")
        else:
            print(f"\n❌ HTML 太短或为空: {len(html)} 字符")
            print(f"完整内容:\n{html[:1000]}")
    except Exception as e:
        print(f"\n❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
