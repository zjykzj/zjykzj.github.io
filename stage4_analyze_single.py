# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/27 16:40
@File    : stage4_analyze_single.py
@Author  : zj
@Description:

单篇文章分类 & 标签分析工具（依赖已生成的 .ontology/ 缓存）

用法:
  python stage4_analyze_single.py .tmp/xxx.json

要求:
  - .ontology/tags.json 必须存在
  - .ontology/categories.json 必须存在
  - .ontology/category_schema.json 必须存在（用于判断主类是否可细分）

输出:
  - 打印该文章的标准化标签和分配的 [主类, 子类] 列表（最多3个）
  - 不修改任何文件
"""

import json
from pathlib import Path
import sys
import openai
import time

# === 配置 ===
OUTPUT_DIR = Path(".ontology")
MODEL_NAME = "deepseek-chat"
openai.base_url = "https://api.deepseek.com/v1/"
openai.api_key = "sk-"


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    for attempt in range(3):
        try:
            resp = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠️ 重试 {attempt + 1}/3: {e}, 等待 {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM 调用失败")


def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.json'):
        print("用法: python stage4_analyze_single.py .tmp/xxx.json", file=sys.stderr)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}", file=sys.stderr)
        sys.exit(1)

    # 检查必要缓存
    tags_file = OUTPUT_DIR / "tags.json"
    cats_file = OUTPUT_DIR / "categories.json"
    schema_file = OUTPUT_DIR / "category_schema.json"

    for f in [tags_file, cats_file, schema_file]:
        if not f.exists():
            print(f"❌ 缺少必要缓存文件: {f}", file=sys.stderr)
            print("请先运行 `python stage2_build_ontology.py` 生成全局体系。")
            sys.exit(1)

    # 加载文章元数据
    with open(json_path, 'r', encoding='utf-8') as fp:
        meta = json.load(fp)
        if 'error' in meta:
            print(f"❌ 文章解析失败: {meta['error']}", file=sys.stderr)
            sys.exit(1)

    stem = json_path.stem
    print(f"🔍 分析文章: {json_path.name}")

    # 加载缓存
    with open(tags_file, 'r', encoding='utf-8') as f:
        standardized_tags = json.load(f)["tags"]
    with open(cats_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)["categories"]
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    # 构建 alias -> standard_tag 映射
    alias_to_std = {}
    for item in standardized_tags:
        std = item.get("standard_tag", "").strip()
        if std:
            alias_to_std[std] = std
            for alias in item.get("aliases", []):
                a = alias.strip()
                if a:
                    alias_to_std[a] = std

    # 构建合法路径集合
    main_defs = {cat["name"]: cat for cat in schema["main_categories"]}
    VALID_MAINS = set(main_defs.keys())
    NON_SUBDIVIDABLE = {name for name, d in main_defs.items() if not d["allow_subcategory"]}

    valid_paths = set()
    for cat in categories:
        path = cat.get("path")
        if not isinstance(path, list) or not (1 <= len(path) <= 2):
            continue
        main = path[0]
        if main not in VALID_MAINS:
            continue
        if main in NON_SUBDIVIDABLE and len(path) != 1:
            continue
        if main not in NON_SUBDIVIDABLE and len(path) != 2:
            continue
        valid_paths.add(tuple(path))

    # 标准化标签
    raw_tags = meta.get("tags", [])
    final_tags = []
    for rt in raw_tags:
        rt_clean = str(rt).strip()
        if rt_clean:
            final_tags.append(alias_to_std.get(rt_clean, rt_clean))

    # 准备 LLM 分配
    idea = meta.get("main_idea", "").strip() or "（无主旨）"
    type_hint = meta.get("content_type", "").strip() or "（未知类型）"
    tags_str = ", ".join(final_tags[:10]) if final_tags else "（无标签）"

    display_paths = sorted(valid_paths, key=lambda p: (p[0], len(p), p[1] if len(p) > 1 else ""))
    cat_options = "\n".join(f"- {list(p)}" for p in display_paths)

    prompt = f"""你是一位严谨的博客分类专家。请为以下文章从**下方明确列出的路径中**选择最相关的分类。

📌 要求：
- 最多选择 3 个分类路径；
- **必须严格使用“可用分类路径”中的条目**；
- 路径格式说明：
  • 单层：["人生感悟"] —— 仅用于不可细分主类
  • 双层：["踩坑记录", "CUDA配置"] —— 用于其余所有主类
- 如果没有相关项，返回空列表。

可用分类路径（共 {len(display_paths)} 条）：
{cat_options}

文章信息：
- 写作类型: {type_hint}
- 主旨: {idea}
- 相关标签: {tags_str}

输出格式（仅 JSON）：
{{
  "categories": [
    ["主类"],
    ["主类", "子类"],
    ...
  ]
}}
"""

    final_cats = []
    try:
        output = call_llm(prompt, temperature=0.0)
        if output.startswith("```json"):
            output = output[7:-3].strip()
        elif output.startswith("```"):
            output = output[3:-3].strip()
        data = json.loads(output)
        candidate_cats = data.get("categories", [])
        if isinstance(candidate_cats, list):
            for c in candidate_cats:
                if isinstance(c, list) and tuple(c) in valid_paths:
                    final_cats.append(c)
                    if len(final_cats) >= 3:
                        break
    except Exception as e:
        print(f"⚠️ 分类分配失败，返回空类别: {e}", file=sys.stderr)
        final_cats = []

    # ========== 修改：严格对齐 Hexo front-matter 的 YAML 缩进风格 ==========
    print("\n" + "=" * 50)
    print("📄 Front-matter 兼容输出:")
    print(f"# 文件: {stem}.md")

    # categories: 使用两层列表，每行 - 前加 2 空格（顶级），子类再缩进 2 空格
    print("\ncategories:")
    if final_cats:
        for cat in final_cats:
            print(f"  - - {cat[0]}")
            if len(cat) > 1:
                print(f"    - {cat[1]}")
    else:
        print("  []")

    # tags: 每个标签前统一 2 空格 + -
    unique_tags = sorted(set(final_tags))  # 可选：保留原序可去掉 sorted
    print("\ntags:")
    if unique_tags:
        for tag in unique_tags:
            print(f"  - {tag}")
    else:
        print("  []")

    print("=" * 50)


if __name__ == "__main__":
    main()