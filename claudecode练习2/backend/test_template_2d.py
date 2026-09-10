"""测试 template-2d 模式：提取 → 场景 → lesson JSON 2D"""
import os, sys, json, asyncio, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import extract_text
from script_generator import generate_script, generate_lesson_json_2d


async def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = None
    for ext in ['.pptx', '.docx', '.pdf']:
        candidate = os.path.join(test_dir, 'test_sample' + ext)
        if os.path.exists(candidate):
            test_file = candidate
            break
    if not test_file:
        print("❌ 未找到测试文件")
        return

    print(f"📄 {os.path.basename(test_file)}")

    t0 = time.time()
    text = extract_text(test_file)
    print(f"[1/3] 文本提取 — {len(text)} 字符, {time.time()-t0:.0f}s")

    t0 = time.time()
    scenes = (await generate_script(text)).model_dump()["scenes"]
    print(f"[2/3] 场景拆分 — {len(scenes)} 个场景, {time.time()-t0:.0f}s")

    t0 = time.time()
    lesson = await generate_lesson_json_2d(scenes, original_text=text)
    elapsed = time.time() - t0
    print(f"[3/3] 2D lesson JSON — {elapsed:.0f}s")

    # 验证
    errors = []
    if 'meta' not in lesson: errors.append("missing meta")
    if 'scene' not in lesson: errors.append("missing scene")
    if 'steps' not in lesson: errors.append("missing steps")

    if not errors:
        print(f"\n✅ 验证通过")
        print(f"  scene.type: {lesson['scene'].get('type')}")
        print(f"  steps: {len(lesson['steps'])}")
        for s in lesson['steps']:
            print(f"    {s.get('title')}")
        path = os.path.join(test_dir, '..', 'frontend', 'test_lesson_2d.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(lesson, f, ensure_ascii=False, indent=2)
        print(f"📁 已保存: {os.path.abspath(path)}")
    else:
        print(f"❌ {errors}")
        print(json.dumps(lesson, ensure_ascii=False, indent=2)[:1000])


if __name__ == '__main__':
    asyncio.run(main())
