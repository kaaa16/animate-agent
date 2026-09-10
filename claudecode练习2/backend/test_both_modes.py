"""测试 2D 和 3D 两个模式（模板优先，失败降级）"""
import requests, json, time, os

test_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(test_dir, 'test_sample.pptx')

for mode in ['3d', '2d']:
    print(f"--- mode={mode} ---")
    t0 = time.time()
    with open(test_file, 'rb') as f:
        r = requests.post(
            f"http://localhost:8000/api/generate?mode={mode}",
            files={"file": f},
            timeout=600
        )
    elapsed = time.time() - t0
    data = r.json()
    has_lesson = "lesson_json" in data
    has_html = "animation_html" in data
    fb = data.get("fallback", False)
    print(f"  {elapsed:.0f}s | status={r.status_code} | lesson_json={has_lesson} | animation_html={has_html} | fallback={fb}")
    if has_lesson:
        st = data["lesson_json"].get("scene", {}).get("type", "?")
        print(f"  scene_type: {st}")
    if fb:
        print(f"  reason: {data.get('fallback_reason', '?')}")
    print()
print("Done")
