"""全流程 API 测试（template 模式）"""
import requests
import json
import time
import os

test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_sample.pptx')
if not os.path.exists(test_file):
    print(f"Test file not found: {test_file}")
    exit(1)

print(f"Testing with: {os.path.basename(test_file)}")
t0 = time.time()

with open(test_file, 'rb') as f:
    files = {'file': f}
    r = requests.post(
        'http://localhost:8000/api/generate?mode=template',
        files=files,
        timeout=600
    )

elapsed = time.time() - t0
print(f"Status: {r.status_code}")
print(f"Total time: {elapsed:.1f}s")

if r.status_code == 200:
    data = r.json()
    print(f"\nmode: {data['mode']}")
    print(f"scenes: {len(data['scenes'])}")
    print(f"Has animation_html: {'animation_html' in data}")

    lj = data.get('lesson_json', {})
    print(f"\n--- lesson.json ---")
    print(f"meta.title: {lj.get('meta', {}).get('title', 'N/A')}")
    print(f"scene.type: {lj.get('scene', {}).get('type', 'N/A')}")
    print(f"steps: {len(lj.get('steps', []))}")

    for i, s in enumerate(lj.get('steps', [])):
        print(f"  [{i+1}] {s.get('title', '?')}: {s.get('description', '')[:60]}...")

    print("\n✅ Full pipeline test PASSED")
else:
    print(f"Error: {r.text[:500]}")
