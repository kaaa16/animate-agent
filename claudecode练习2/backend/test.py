"""
本地调试脚本：测试 parser.py + script_generator.py。
用法：
  python test.py parse <文件路径>        — 仅提取文本
  python test.py generate <文件路径>    — 提取文本 → AI 生成脚本 JSON
示例：
  python test.py parse test_sample.pptx
  python test.py generate test_sample.pptx
"""

"""
本地调试脚本：测试 parser.py + script_generator.py。
用法：
  python test.py parse <文件路径>           — 仅提取文本
  python test.py generate <文件路径>        — 提取文本 → AI 生成场景 JSON
  python test.py animation <文件路径>       — 全流程：文本 → 场景 → 动画 HTML
示例：
  python test.py parse test_sample.pptx
  python test.py generate test_sample.pptx
  python test.py animation test_sample.pptx
"""

import sys
import os
import json
from parser import extract_text


def cmd_parse(path: str):
    """提取文件纯文本"""
    text = extract_text(path)
    print(text)
    print(f"\n[完成] 共 {len(text)} 个字符")


def cmd_generate(path: str):
    """提取文本 → 双模型生成教学脚本"""
    text = extract_text(path)
    print(f"[提取] {len(text)} 个字符\n")

    from script_generator import generate_script_sync
    result = generate_script_sync(text)

    print("\n========== 生成的脚本 JSON ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[完成] 共 {len(result)} 个场景")


def cmd_animation(path: str):
    """提取文本 → 双模型生成场景 → Kimi 生成动画 HTML"""
    text = extract_text(path)
    print(f"[提取] {len(text)} 个字符\n")

    from script_generator import generate_script_sync, generate_animation_html_sync
    scenes = generate_script_sync(text)
    print(f"[场景] 共 {len(scenes)} 个场景")

    print("[动画] 正在调用 Kimi 生成动画 HTML...")
    html = generate_animation_html_sync(scenes)

    # 保存到文件
    out_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "generated_animation.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[完成] 动画 HTML 已保存到: {out_path}")
    print(f"[大小] {len(html)} 个字符")


def main():
    if len(sys.argv) < 3:
        print("用法:")
        print("  python test.py parse <文件路径>          — 仅提取文本")
        print("  python test.py generate <文件路径>       — 文本 → AI 场景 JSON")
        print("  python test.py animation <文件路径>      — 全流程：文本 → 场景 → 动画 HTML")
        return

    cmd = sys.argv[1]
    path = sys.argv[2]

    if not os.path.exists(path):
        print(f"[错误] 文件不存在: {path}")
        return

    if cmd == "parse":
        cmd_parse(path)
    elif cmd == "generate":
        cmd_generate(path)
    elif cmd == "animation":
        cmd_animation(path)
    else:
        print(f"[错误] 未知命令: {cmd}")
        print("可用命令: parse  generate  animation")


if __name__ == "__main__":
    main()
