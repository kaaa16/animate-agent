"""验证新场景类型：AI 能否为物理题正确选中 projectile_motion / force_analysis"""
import asyncio, json
from script_generator import generate_lesson_json, generate_lesson_json_2d

# 模拟教学场景（跳过 Kimi/DeepSeek 场景拆分，直接测 lesson JSON 生成）
physics_scenes = [
    {
        "scene": 1,
        "title": "什么是斜抛运动",
        "narration": "斜抛运动是物体以一定初速度、与水平方向成一定角度抛出，在重力作用下沿抛物线运动。",
        "key_points": ["斜抛运动", "抛物线", "初速度"]
    },
    {
        "scene": 2,
        "title": "射程与最大高度",
        "narration": "抛体的射程由初速度和角度决定，最大高度出现在竖直速度为零的时刻。",
        "key_points": ["射程", "最大高度", "角度"]
    },
]

force_scenes = [
    {
        "scene": 1,
        "title": "物体受力分析",
        "narration": "静止在水平面上的物体受到重力和支持力，二力平衡。",
        "key_points": ["重力", "支持力", "平衡"]
    },
]

async def main():
    print("=== 测试 3D 抛物运动 ===")
    lesson = await generate_lesson_json(physics_scenes, original_text="斜抛运动：物体以初速度斜向上抛出，在重力作用下做抛物线运动，射程和最大高度由初速度和角度决定。", mode="3d")
    st = lesson.get("scene", {}).get("type", "?")
    print(f"  scene.type = {st}")
    print(f"  steps = {len(lesson.get('steps', []))}")
    if st == "projectile_motion":
        print("  ✅ 正确选中 projectile_motion")
    else:
        print(f"  ⚠ 未选中 projectile_motion（实际 {st}）")

    print("\n=== 测试 3D 受力分析 ===")
    lesson2 = await generate_lesson_json(force_scenes, original_text="静止物体受力分析：重力、支持力二力平衡。", mode="3d")
    st2 = lesson2.get("scene", {}).get("type", "?")
    print(f"  scene.type = {st2}")
    if st2 == "force_analysis":
        print("  ✅ 正确选中 force_analysis")
    else:
        print(f"  ⚠ 未选中 force_analysis（实际 {st2}）")

    print("\n=== 测试 2D 抛物运动 ===")
    lesson3 = await generate_lesson_json_2d(physics_scenes, original_text="斜抛运动：物体以初速度斜向上抛出，在重力作用下做抛物线运动，射程和最大高度由初速度和角度决定。")
    st3 = lesson3.get("scene", {}).get("type", "?")
    print(f"  scene.type = {st3}")
    if st3 == "projectile_motion_2d":
        print("  ✅ 正确选中 projectile_motion_2d")
    else:
        print(f"  ⚠ 未选中 projectile_motion_2d（实际 {st3}）")

asyncio.run(main())
