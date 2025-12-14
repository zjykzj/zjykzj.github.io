# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/14 16:54
@File    : assign_categories_tags.py
@Author  : zj
@Description: 

第二阶段：为博客文章智能分配分类（categories）与标签（tags）

功能说明：
- 基于预定义的 categories.yaml 和 tag_vocabulary.yaml 体系，
  调用 DeepSeek 大模型为每篇 Markdown 博客文章生成语义匹配的分类路径与技术标签；
- 默认限制：每篇文章最多分配 3 条分类路径、8 个标签，确保结构简洁；
- 支持两种运行模式：
    • 批量模式：处理整个文章目录；
    • 单文件模式：针对特定文章（如综述、年度总结）单独处理，并可临时放宽数量限制；
- 自动检测并报告使用了新类别或新标签的文章，便于人工审核；
- 内置强重试机制（最多 5 次）与详细的 token/耗时统计。

输入：
  - 文章目录（或单个 .md 文件）
  - categories.yaml：预定义的分类路径列表（如 ["AI", "大模型"]）
  - tag_vocabulary.yaml：预定义的标准标签词表

输出：
  - 直接修改每篇 .md 文件的 front-matter，写入 categories 和 tags 字段
  - 控制台输出处理报告：含新项文章列表、总耗时、总 token 消耗等

典型使用方式：

# 1. 批量处理（默认配置：最多 3 个分类、8 个标签，读取前 3000 字符）
python assign_categories_tags.py ./source/_posts

# 2. 单独处理某篇长文（如年度总结），使用全文并放宽限制
python assign_categories_tags.py \
  --single-file ./source/_posts/2024-year-in-review.md \
  --max-categories 5 \
  --max-tags 15 \
  --max-content-chars -1

# 3. 批量处理但略微放宽数量限制（谨慎使用，避免体系污染）
python assign_categories_tags.py ./source/_posts \
  --max-categories 4 \
  --max-tags 10

# 4. 自定义输入长度（例如只看前 5000 字符）
python assign_categories_tags.py ./source/_posts --max-content-chars 5000

注意：
- 必须设置环境变量 OPENAI_API_KEY 或在代码中配置 DeepSeek API 密钥；
- 预定义体系文件路径可通过 --categories 和 --tags 参数指定。
"""

import os
import yaml
import argparse
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple

import frontmatter  # pip install python-frontmatter
import openai  # 确保已设置 OPENAI_API_KEY 和 base_url

# ======================
# 配置
# ======================
TEMPERATURE = 0.0
MAX_RETRIES = 5  # 强制成功，最多重试 5 次

# DeepSeek API 配置（兼容 OpenAI）
MODEL_NAME = "deepseek-reasoner"
os.environ["OPENAI_API_KEY"] = "sk-"
openai.base_url = "https://api.deepseek.com/v1/"
openai.api_key = os.getenv("OPENAI_API_KEY")

# 全局统计
TOTAL_STATS = {
    "total_time": 0.0,
    "total_requests": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
}


# ======================
# 工具函数
# ======================

def load_predefined_sets(categories_yaml: str, tags_yaml: str) -> Tuple[Set[Tuple], Set[str]]:
    with open(categories_yaml, 'r', encoding='utf-8') as f:
        cat_data = yaml.safe_load(f)
    with open(tags_yaml, 'r', encoding='utf-8') as f:
        tag_data = yaml.safe_load(f)

    valid_cat_paths = set(tuple(c["path"]) for c in cat_data["categories"])
    valid_tags = set(t["standard_tag"] for t in tag_data["tags"])
    return valid_cat_paths, valid_tags


def collect_markdown_files(posts_dir: str) -> List[Path]:
    return sorted(Path(posts_dir).rglob("*.md"))


def read_article(file_path: Path, max_content_chars: int = 3000) -> Dict:
    """
    读取 Markdown 文章，提取标题和内容预览。

    Args:
        file_path: 文章路径
        max_content_chars: 最大读取字符数；若为 -1，则读取全文

    Returns:
        dict: 包含 file, title, preview 的字典
    """
    post = frontmatter.load(str(file_path))
    title = post.get('title', file_path.stem)
    content = post.content

    if max_content_chars == -1:
        # 读入整篇文章
        preview = " ".join(content.split())  # 标准化空白
    else:
        # 截断并标准化
        preview = " ".join(content[:max_content_chars].split())

    return {
        "file": file_path,
        "title": title,
        "preview": preview
    }


def build_prompt(
        article: Dict,
        all_categories: List[List[str]],
        all_tags: List[str],
        max_categories: int = 3,
        max_tags: int = 8
) -> str:
    # 可读格式
    readable_cats = ["[" + ", ".join(f'"{part}"' for part in path) + "]" for path in all_categories]
    cats_str = "\n".join([f"- {c}" for c in readable_cats])
    tags_str = "\n".join([f"- {t}" for t in all_tags])

    return f"""你是一位博客内容分类专家。请为以下文章分配最合适的分类路径和相关技术标签。

    要求：
    1. 分类（categories）：
       - 从“可用分类路径”中选择 **1 到 {max_categories} 条最相关的路径**
       - 每条路径是一个列表，如 ["大类", "子类"]
       - 多条路径时，输出为列表的列表（见下方示例）
    2. 标签（tags）：
       - 从“可用标签”中选择 **1 到 {max_tags} 个**最相关的标准标签
    3. 如果文章涉及全新领域，可返回你认为合理的路径或标签（我们会人工审核）
    4. 输出必须是纯 YAML，仅包含 categories 和 tags 两个字段，不要任何额外内容

    ---
    可用分类路径（每项是一条完整路径）：
    {cats_str}

    ---
    可用标签：
    {tags_str}

    ---
    文章标题: {article['title']}
    内容预览:
    {article['preview']}

    ---
    输出示例（多条分类路径）：
    categories:
      - ["技术教程", "Python"]
      - ["工程实践", "部署"]
    tags:
      - Docker
      - Linux
      - CI/CD
    """


def call_llm_with_stats(prompt: str) -> Dict:
    global TOTAL_STATS
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start_time = time.time()
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=65536,
            )
            duration = time.time() - start_time

            # 提取 token 信息
            usage = response.usage
            prompt_tk = usage.prompt_tokens
            completion_tk = usage.completion_tokens

            # 更新全局统计
            TOTAL_STATS["total_requests"] += 1
            TOTAL_STATS["total_prompt_tokens"] += prompt_tk
            TOTAL_STATS["total_completion_tokens"] += completion_tk
            TOTAL_STATS["total_time"] += duration

            print(f"⏱️  LLM Call | 耗时: {duration:.2f}s | 输入: {prompt_tk} tk | 输出: {completion_tk} tk")

            # 解析输出
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            data = yaml.safe_load(text)

            return {
                "categories": data.get("categories", []),
                "tags": [str(t).strip() for t in data.get("tags", []) if t]
            }

        except Exception as e:
            last_error = e
            wait_time = min(2 ** attempt, 10)
            print(f"⚠️  第 {attempt}/{MAX_RETRIES} 次调用失败: {e}. 等待 {wait_time}s...")
            time.sleep(wait_time)

    # 所有重试失败
    raise RuntimeError(f"LLM 调用彻底失败（{MAX_RETRIES} 次重试后仍失败）: {last_error}")


def normalize_categories(raw_cats, max_paths: int = 3):
    """将 LLM 返回的 categories 统一转为 List[List[str]]，最多保留 max_paths 条路径"""
    if not raw_cats:
        return []

    # 确保是列表类型
    if not isinstance(raw_cats, list):
        return []

    # 情况1: ["AI", "大模型"] → 单条路径（扁平列表）
    if len(raw_cats) > 0 and isinstance(raw_cats[0], str):
        # 合并为一条路径（即使 max_paths > 1，也只有一条）
        return [list(raw_cats)]

    # 情况2: [["AI", "大模型"], ["工程", "部署"]] → 多条路径
    if len(raw_cats) > 0 and isinstance(raw_cats[0], list):
        # 只取前 max_paths 条，每条转为 list[str]
        return [list(path) for path in raw_cats[:max_paths] if isinstance(path, list)]

    # 其他异常格式（如混合、None 等）
    return []


def write_frontmatter(file_path: Path, new_cats, new_tags: List[str]):
    # 不指定 new_cats 类型，因为可能是 List[str] 或 List[List[str]]
    post = frontmatter.load(str(file_path))
    post['categories'] = new_cats
    post['tags'] = new_tags
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))


# ======================
# 主逻辑
# ======================

def main(
        posts_dir: str = None,
        single_file: str = None,
        categories_yaml: str = "categories.yaml",
        tags_yaml: str = "tag_vocabulary.yaml",
        max_categories: int = 3,
        max_tags: int = 8,
        max_content_chars: int = 3000,  # ← 新增参数
):
    print("🔍 加载预定义体系...")
    valid_cat_paths, valid_tags = load_predefined_sets(categories_yaml, tags_yaml)
    all_cat_list = [list(p) for p in valid_cat_paths]
    all_tag_list = list(valid_tags)
    print(f"✅ 共 {len(all_cat_list)} 个分类路径，{len(all_tag_list)} 个标准标签")

    # 确定要处理的文件列表
    if single_file:
        file_path = Path(single_file)
        if not file_path.exists():
            raise FileNotFoundError(f"指定的单个文件不存在: {file_path}")
        md_files = [file_path]
        print(f"🎯 单文件模式: {file_path.name}")
    elif posts_dir:
        md_files = collect_markdown_files(posts_dir)
        print(f"📝 批量模式: 共找到 {len(md_files)} 篇文章\n")
    else:
        raise ValueError("必须指定 posts_dir 或 single_file")

    articles_with_new_items = []

    for i, file_path in enumerate(md_files, 1):
        if single_file:
            print(f"\n🎯 处理单篇文章: {file_path.name}")
        else:
            print(f"\n[{i}/{len(md_files)}] 处理: {file_path.name}")

        try:
            # ← 传入 max_content_chars
            art = read_article(file_path, max_content_chars=max_content_chars)
            prompt = build_prompt(
                art,
                all_cat_list,
                all_tag_list,
                max_categories=max_categories,
                max_tags=max_tags
            )
            result = call_llm_with_stats(prompt)

            assigned_cats = normalize_categories(result.get("categories"), max_paths=max_categories)
            assigned_tags_raw = result.get("tags", [])
            assigned_tags = [str(t).strip() for t in assigned_tags_raw if t][:max_tags]

            # 检查新类别/标签
            has_new_cat = any(
                isinstance(path, list) and tuple(path) not in valid_cat_paths
                for path in assigned_cats
            )
            new_tags_in_result = [t for t in assigned_tags if t not in valid_tags]
            has_new_tag = len(new_tags_in_result) > 0

            if has_new_cat or has_new_tag:
                articles_with_new_items.append({
                    "file": str(file_path),
                    "title": art["title"],
                    "new_categories": assigned_cats if has_new_cat else None,
                    "new_tags": new_tags_in_result
                })

            write_frontmatter(file_path, assigned_cats, assigned_tags)
            print(f"  → 写入 categories: {assigned_cats}")
            print(f"  → 写入 tags: {assigned_tags}")

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            continue

    # === 报告 ===
    print("\n" + "=" * 60)
    print("📊 全局统计:")
    print(f"  • 总耗时: {TOTAL_STATS['total_time']:.2f} 秒")
    print(f"  • 总请求数: {TOTAL_STATS['total_requests']}")
    print(f"  • 总输入 tokens: {TOTAL_STATS['total_prompt_tokens']}")
    print(f"  • 总输出 tokens: {TOTAL_STATS['total_completion_tokens']}")
    print("=" * 60)

    if articles_with_new_items:
        print(f"\n⚠️  发现 {len(articles_with_new_items)} 篇文章使用了新的类别或标签，需人工审核：\n")
        for item in articles_with_new_items:
            print(f"📄 {item['file']}")
            if item['new_categories']:
                print(f"   新类别: {item['new_categories']}")
            if item['new_tags']:
                print(f"   新标签: {item['new_tags']}")
            print()
    else:
        print("\n✅ 所有文章均使用了预定义体系，无需人工干预。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为博客文章分配分类和标签（支持批量或单文件）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("posts_dir", nargs="?", help="博客文章目录（批量模式）")
    group.add_argument("--single-file", type=str, help="单个 Markdown 文件路径（单文件模式）")

    parser.add_argument("--categories", default="categories.yaml", help="分类定义文件")
    parser.add_argument("--tags", default="tag_vocabulary.yaml", help="标签词表文件")
    parser.add_argument("--max-categories", type=int, default=3, help="最多分类路径数（默认: 3）")
    parser.add_argument("--max-tags", type=int, default=8, help="最多标签数（默认: 8）")
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=3000,
        help="每篇文章最大读取字符数（默认: 3000；设为 -1 表示读取全文）"
    )

    args = parser.parse_args()

    main(
        posts_dir=args.posts_dir,
        single_file=args.single_file,
        categories_yaml=args.categories,
        tags_yaml=args.tags,
        max_categories=args.max_categories,
        max_tags=args.max_tags,
        max_content_chars=args.max_content_chars,  # ← 传入新参数
    )
