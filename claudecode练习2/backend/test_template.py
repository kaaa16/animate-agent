"""
测试模板驱动方案：提取文本 → 场景拆分 → lesson JSON 生成
"""
import os
import sys
import json
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import extract_text
from script_generator import generate_script, generate_lesson_json


async def main():
    # 找测试文件
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = None
    for ext in ['.pptx', '.docx', '.pdf']:
        candidate = os.path.join(test_dir, 'test_sample' + ext)
        if os.path.exists(candidate):
            test_file = candidate
            break

    if not test_file:
        print("❌ 未找到测试文件 (test_sample.pptx/docx/pdf)")
        return

    print(f"📄 测试文件: {os.path.basename(test_file)}")
    print("=" * 60)

    # Step 1: 提取文本
    t0 = time.time()
    text = extract_text(test_file)
    print(f"[1/3] 文本提取完成 — {len(text)} 字符, 耗时 {time.time()-t0:.1f}s")
    print(f"  文本预览: {text[:200]}...")

    # Step 2: 生成场景
    t0 = time.time()
    scenes = (await generate_script(text)).model_dump()["scenes"]
    print(f"[2/3] 场景拆分完成 — {len(scenes)} 个场景, 耗时 {time.time()-t0:.1f}s")
    for s in scenes:
        print(f"  场景 {s['scene']}: {s['title']}")

    # Step 3: 生成 lesson JSON
    t0 = time.time()
    lesson = await generate_lesson_json(scenes, original_text=text)
    elapsed = time.time() - t0
    print(f"[3/3] lesson JSON 生成完成 — 耗时 {elapsed:.1f}s")

    # 验证
    print("\n📋 验证结果:")
    errors = []

    if 'meta' not in lesson:
        errors.append("缺少 meta")
    else:
        print(f"  meta.title: {lesson['meta'].get('title', '[缺失]')}")

    if 'scene' not in lesson:
        errors.append("缺少 scene")
    else:
        print(f"  scene.type: {lesson['scene'].get('type', '[缺失]')}")

    if 'steps' not in lesson or not isinstance(lesson['steps'], list):
        errors.append("steps 缺失或格式错误")
    else:
        print(f"  steps: {len(lesson['steps'])} 个步骤")
        for i, step in enumerate(lesson['steps']):
            print(f"    Step {i+1}: {step.get('title', '[无标题]')}")
            ostates = step.get('objectStates', {})
            print(f"      objectStates keys: {list(ostates.keys()) if ostates else '[空]'}")

    if errors:
        print(f"\n❌ 验证失败: {', '.join(errors)}")
        # 打印原始 JSON 帮助调试
        print("\n--- 原始输出 ---")
        print(json.dumps(lesson, ensure_ascii=False, indent=2)[:2000])
    else:
        print("\n✅ 验证通过！")

        # 写入文件供播放器测试
        output_path = os.path.join(test_dir, '..', 'frontend', 'test_lesson.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lesson, f, ensure_ascii=False, indent=2)
        print(f"📁 lesson JSON 已保存到: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    asyncio.run(main())
