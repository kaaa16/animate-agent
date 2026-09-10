"""API 全流程测试 template-2d"""
import requests, json, time, os

test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_sample.pptx')

print(f"Testing: template-2d")
t0 = time.time()

with open(test_file, 'rb') as f:
    r = requests.post('http://localhost:8000/api/generate?mode=template-2d',
                      files={'file': f}, timeout=600)

elapsed = time.time() - t0
print(f"Status: {r.status_code}, Time: {elapsed:.0f}s")

if r.status_code == 200:
    data = r.json()
    lj = data.get('lesson_json', {})
    print(f"mode: {data['mode']}")
    print(f"scene.type: {lj.get('scene', {}).get('type')}")
    print(f"steps: {len(lj.get('steps', []))}")
    for s in lj.get('steps', []):
        print(f"  [{s['id']}] {s['title']}")
    print("\n✅ template-2d API test PASSED")
else:
    print(f"FAIL: {r.text[:300]}")
