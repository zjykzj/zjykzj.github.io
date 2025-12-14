# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/14
@File    : generate_ontology.py
@Author  : zj
@Description:

第一阶段：自动生成博客的标准化分类体系（categories.yaml）与标签词表（tag_vocabulary.yaml）

功能说明：
- 基于现有博客文章的内容、标题及已有 front-matter 信息，
  利用大模型（DeepSeek）自动归纳出一套**结构化、可维护、语义一致**的预定义分类与标签体系；
- **分类（Categories）聚焦文章类型/目的**（如“教程 > 工具使用”、“复盘 > 项目总结”），而非技术主题；
- **标签（Tags）聚焦具体技术实体**（如 Docker、YOLOv5、CI/CD），并自动合并别名（aliases）；
- 支持大规模文章处理：标签生成采用分批策略（默认每批 50 篇），避免上下文过长；
- 内置强健容错机制：
    • LLM 调用自动重试（最多 3~5 次）
    • YAML 输出解析失败时自动纠错重试
    • 异常批次保存原始调试文件（.raw）
- 全程记录 token 消耗、请求次数与耗时，便于成本评估。

输入：
  - 博客文章目录（默认: blog/source/_posts）
  - 每篇文章需为 Markdown 格式，可含现有 categories/tags（用于参考）

输出：
  - categories.yaml：标准化分类体系（含 path / description / matching_hints）
  - tag_vocabulary.yaml：标准化标签词表（含 standard_tag / aliases）
  - （可选）失败批次的 .raw 调试文件

典型使用方式：

# 直接运行（使用默认配置）
python generate_ontology.py

# 注意：
# - 本脚本通常只需在博客体系初期或重大重构时运行一次；
# - 生成的 YAML 文件**必须经过人工审核和调整**后再用于第二阶段（assign_categories_tags.py）；
# - 如需处理不同博客目录，请修改代码中的 BLOG_POSTS_DIR 变量，或将其改为命令行参数（当前为硬编码）。

依赖：
  - Python 包：pyyaml, python-frontmatter, openai
  - DeepSeek API 密钥（已内置，建议改用环境变量管理）

此脚本是博客元数据自动化 pipeline 的第一步，为后续智能分类打下结构化基础。
"""

import os
import re
import yaml
import time
from pathlib import Path
from typing import Any, Dict, List
from frontmatter import load as load_frontmatter
import openai

# ======================
# 配置区（请按需修改）
# ======================
BLOG_POSTS_DIR = Path("blog/source/_posts")  # 博文根目录
OUTPUT_CATEGORIES = "categories.yaml"
OUTPUT_TAGS = "tag_vocabulary.yaml"

# DeepSeek API 配置（兼容 OpenAI）
MODEL_NAME = "deepseek-reasoner"
os.environ["OPENAI_API_KEY"] = "sk-"
openai.base_url = "https://api.deepseek.com/v1/"
openai.api_key = os.getenv("OPENAI_API_KEY")

# 全局统计变量
TOTAL_STATS = {
    "total_time": 0.0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_requests": 0,
}


# ======================
# 工具函数
# ======================
def extract_lead_or_preview(content: str, max_chars=2000) -> str:
    if "<!-- more -->" in content:
        lead = content.split("<!-- more -->")[0].strip()
    else:
        lead = content[:max_chars]
    return lead.strip()


def clean_markdown_text(md_text: str) -> str:
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]*`", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md_text)
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md_text)
    md_text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", md_text)
    md_text = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE)
    return md_text.strip()


def collect_articles() -> List[Dict]:
    articles = []
    for md_file in BLOG_POSTS_DIR.rglob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                post = load_frontmatter(f)
                content = post.content
                front_matter = dict(post)
                title = front_matter.get("title", md_file.stem)

                # 处理 categories
                raw_cats = front_matter.get("categories", [])
                if isinstance(raw_cats, str):
                    existing_categories = [raw_cats]
                elif isinstance(raw_cats, list):
                    def flatten(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, list):
                                result.extend(flatten(item))
                            else:
                                result.append(str(item).strip())
                        return result

                    existing_categories = flatten(raw_cats)
                else:
                    existing_categories = [str(raw_cats)]

                # 处理 tags
                raw_tags = front_matter.get("tags", [])
                if isinstance(raw_tags, str):
                    existing_tags = [raw_tags]
                elif isinstance(raw_tags, list):
                    def flatten(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, list):
                                result.extend(flatten(item))
                            else:
                                result.append(str(item).strip())
                        return result

                    existing_tags = flatten(raw_tags)
                else:
                    existing_tags = [str(raw_tags)]

                preview = extract_lead_or_preview(content, max_chars=2000)
                full_text = clean_markdown_text(content)

                articles.append({
                    "file": str(md_file),
                    "title": title,
                    "preview": preview,
                    "full_text": full_text,
                    "front_matter": front_matter,
                    "existing_categories": existing_categories,
                    "existing_tags": existing_tags,
                })
        except Exception as e:
            print(f"⚠️ 解析失败: {md_file} - {e}")
    return articles


def log_llm_call(prompt: str, response: Any, start_time: float, task_name: str = "LLM Call") -> None:
    duration = time.time() - start_time
    usage = getattr(response, 'usage', None)
    prompt_tk = getattr(usage, 'prompt_tokens', 0)
    completion_tk = getattr(usage, 'completion_tokens', 0)

    TOTAL_STATS["total_prompt_tokens"] += prompt_tk
    TOTAL_STATS["total_completion_tokens"] += completion_tk
    TOTAL_STATS["total_requests"] += 1

    print(f"⏱️  {task_name} | 耗时: {duration:.2f}s | "
          f"输入: {prompt_tk} tk | 输出: {completion_tk} tk")


def robust_llm_call(prompt: str, max_retries: int = 3, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            log_llm_call(prompt, response, start_time)
            raw_output = response.choices[0].message.content.strip()

            # 移除可能的 markdown 包裹
            if raw_output.startswith("```yaml"):
                raw_output = raw_output[7:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
            return raw_output.strip()

        except Exception as e:
            last_error = e
            wait_time = min(2 ** attempt, 10)  # 最多等 10 秒
            print(f"⚠️  第 {attempt}/{max_retries} 次调用失败: {e}. 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

    raise RuntimeError(f"LLM 调用失败（{max_retries} 次重试后仍失败）: {last_error}")


# ======================
# 任务 A: 生成 Category 体系
# ======================
def build_category_prompt(articles: List[Dict]) -> str:
    snippets = []
    for art in articles:
        fm = art["front_matter"]
        meta_lines = []
        if fm.get("temporal_type"):
            meta_lines.append(f"temporal_type: {fm['temporal_type']}")
        if fm.get("intent"):
            meta_lines.append(f"intent: {fm['intent']}")
        if art["existing_categories"]:
            meta_lines.append(f"当前分类: {', '.join(art['existing_categories'])}")
        if art["existing_tags"]:
            meta_lines.append(f"当前标签: {', '.join(art['existing_tags'][:5])}")

        meta_str = "\n".join(meta_lines) if meta_lines else "无"

        snippet = (
            f"标题: {art['title']}\n"
            f"内容预览:\n{art['preview'][:2000]}\n"
            f"元信息与现有标注:\n{meta_str}"
        ).strip()
        snippets.append(snippet)

    article_snippets = "\n\n---\n\n".join(snippets)

    return f"""你是一位博客架构师。你的任务是：**基于以下所有文章的现有分类、标签、内容和元信息，设计一个统一、合理、结构清晰的预定义分类体系（Category Hierarchy）**。

背景：
- 每篇文章已有 `categories` 和 `tags`，但可能存在不一致、冗余或粒度不合理的问题。
- 你需要**归纳出一套新的、标准化的 category 体系**，用于未来所有文章的自动分类。

要求：
1. **Category 必须反映文章的写作目的或文体类型**（如年度总结、教程、项目复盘），**不是技术主题**。
   - ✅ 正确示例：["博客", "年度总结"]、["教程", "工具使用"]
   - ❌ 错误示例：["深度学习"]、["计算机视觉"]（这是 tag 的范畴）
2. 使用两级结构：[大类, 子类]
3. 为每个 category 提供：
   - `path`: [大类, 子类]
   - `description`: 一句话说明适用场景
   - `matching_hints`: 3~5 个关键词、信号或规则（用于后续自动匹配）
4. 大类不超过 20 个。

以下是全部文章的信息：
---
{article_snippets}
---

请以 YAML 格式输出，顶层为 `categories`，每个元素包含 path/description/matching_hints。
不要任何解释，不要 markdown 代码块，直接输出 YAML 内容。
"""


def run_task_a(articles: List[Dict]):
    start_time = time.time()
    raw_output = ""
    try:
        prompt = build_category_prompt(articles)
        raw_output = robust_llm_call(prompt, max_retries=3, temperature=0.3, max_tokens=65536)

        data = yaml.safe_load(raw_output)
        with open(OUTPUT_CATEGORIES, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, indent=2, sort_keys=False)
        print(f"✅ 任务 A 完成：已生成 {OUTPUT_CATEGORIES}")
    except Exception as e:
        print(f"❌ 任务 A 最终失败: {e}")
        with open(OUTPUT_CATEGORIES + ".raw", "w", encoding="utf-8") as f:
            f.write(raw_output)
        raise e
    finally:
        task_time = time.time() - start_time
        TOTAL_STATS["total_time"] += task_time
        print(f"📊 任务 A 总耗时: {task_time:.2f} 秒")


# ======================
# 任务 B: 生成 Tag 词表
# ======================
def robust_llm_yaml_call(
        prompt: str,
        max_retries: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 65536,
        expected_type: str = "list"
) -> List[Dict]:
    """
    调用 LLM 并确保返回内容能被解析为合法 YAML，且结构符合预期。
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 调用 LLM
            raw_output = robust_llm_call(
                prompt=prompt,
                max_retries=1,  # 内部不再重试网络错误，由外层控制
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 尝试解析 YAML
            parsed = yaml.safe_load(raw_output)
            if parsed is None:
                raise ValueError("YAML 解析结果为 None")

            # 验证结构
            if expected_type == "list":
                if not isinstance(parsed, list):
                    raise ValueError(f"期望输出为列表，但得到 {type(parsed)}")
                # 简单验证每个元素是 dict 且含 standard_tag
                for item in parsed:
                    if not isinstance(item, dict) or "standard_tag" not in item:
                        raise ValueError("列表中存在非标准条目")
            elif expected_type == "dict_with_tags":
                if not (isinstance(parsed, dict) and "tags" in parsed):
                    raise ValueError("期望字典包含 'tags' 键")
                parsed = parsed["tags"]

            return parsed  # 成功！

        except (yaml.YAMLError, ValueError, TypeError) as e:
            last_error = e
            print(f"⚠️  第 {attempt}/{max_retries} 次 YAML 解析失败: {e}")
            if attempt < max_retries:
                # 可选：在下一次 prompt 中加入纠错指令
                prompt = f"""之前的输出无法被解析为合法 YAML。请严格遵守以下规则重新生成：

- 必须是纯 YAML，无任何解释、标题或 markdown 包裹
- 每个条目格式：
    - standard_tag: 标准名称
      aliases:
        - 别名1
        - 别名2
- 不要使用内联列表如 [a, b]
- 不要省略缩进

现在请重新处理以下内容：
{prompt.split('文章内容如下：')[-1]}"""
                time.sleep(min(2 ** attempt, 8))
            else:
                break

    raise RuntimeError(f"YAML 生成失败（{max_retries} 次重试后仍无效）: {last_error}")


def build_tag_prompt(articles: List[Dict]) -> str:
    snippets = []
    for art in articles:
        short_text = art['full_text'][:2000].replace("\n", " ")
        snippets.append(f"文章: {art['title']}\n内容: {short_text}")
    all_texts = "\n\n---\n\n".join(snippets)
    return f"""你是一位技术术语整理专家。请从以下多篇技术博客中，提取所有**具体、可检索的技术实体**，并构建一个标准化标签词表。

要求：
1. 只提取具体名词：工具、框架、语言、算法、协议、项目名、年份、方法论等。
   - ✅ 例如：Docker, YOLOv5, git, 2019, ResNet, CI/CD, LabelImage
   - ❌ 排除：学习、工作、感觉、提高、问题（太泛）
2. 合并同义词/变体，为每个标准标签列出常见别名。
3. 输出格式为 YAML 列表，每个条目：
   - standard_tag: 标准形式
   - aliases: [变体列表]

文章内容如下：
---
{all_texts}
---

请直接输出 YAML 列表，不要任何解释或 markdown 包裹。
"""


def run_task_b(articles: List[Dict], batch_size: int = 50):
    all_tag_entries = []
    start_time = time.time()

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        batch_index = i // batch_size + 1
        print(f"📦 处理第 {batch_index} 批，共 {len(batch)} 篇文章...")

        prompt = build_tag_prompt(batch)
        try:
            parsed_list = robust_llm_yaml_call(
                prompt=prompt,
                max_retries=5,
                temperature=0.3,
                max_tokens=65536,
                expected_type="list"
            )
            all_tag_entries.extend(parsed_list)
            print(f"✅ 第 {batch_index} 批成功解析 {len(parsed_list)} 个标签")

        except Exception as e:
            # 即使重试 5 次仍失败，记录原始输出供人工检查，但不跳过！
            print(f"💥 第 {batch_index} 批彻底失败（所有重试均无效）: {e}")
            # 保存原始输出用于调试
            raw_debug_file = f"tag_batch_{i}.raw"
            with open(raw_debug_file, "w", encoding="utf-8") as f:
                f.write(prompt)  # 或者你可以保存最后一次 raw_output（需调整函数）
            print(f"   ⚠️ 已保存调试文件: {raw_debug_file}")
            # 注意：这里仍然跳过，因为实在无法解析。但概率极低。
            continue

    # 合并去重
    tag_dict = {}
    for entry in all_tag_entries:
        if not isinstance(entry, dict):
            continue
        std_tag = entry.get("standard_tag")
        aliases = entry.get("aliases", [])
        if not std_tag:
            continue
        std_tag = str(std_tag).strip()
        if std_tag not in tag_dict:
            tag_dict[std_tag] = set()
        for alias in aliases:
            tag_dict[std_tag].add(str(alias).strip())
        tag_dict[std_tag].add(std_tag)

    final_tags = [
        {"standard_tag": std, "aliases": sorted(list(aliases))}
        for std, aliases in tag_dict.items()
    ]

    with open(OUTPUT_TAGS, "w", encoding="utf-8") as f:
        yaml.dump({"tags": final_tags}, f, allow_unicode=True, indent=2, sort_keys=False)

    task_time = time.time() - start_time
    TOTAL_STATS["total_time"] += task_time
    print(f"✅ 任务 B 完成：已生成 {OUTPUT_TAGS}（共 {len(final_tags)} 个标准标签）")
    print(f"📊 任务 B 总耗时: {task_time:.2f} 秒")


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    print("🔍 正在扫描博客文章...")
    articles = collect_articles()
    print(f"📚 共找到 {len(articles)} 篇文章\n")

    try:
        print("🚀 执行任务 A：生成预定义 Category 体系...")
        run_task_a(articles)

        print("\n🚀 执行任务 B：生成标准化 Tag 词表...")
        run_task_b(articles)

        print("\n🎉 第一阶段完成！请检查生成的 YAML 文件并进行人工审核。")
    finally:
        print("\n" + "=" * 60)
        print("📈 全局统计:")
        print(f"  • 总耗时: {TOTAL_STATS['total_time']:.2f} 秒")
        print(f"  • 总请求数: {TOTAL_STATS['total_requests']}")
        print(f"  • 总输入 tokens: {TOTAL_STATS['total_prompt_tokens']}")
        print(f"  • 总输出 tokens: {TOTAL_STATS['total_completion_tokens']}")
        print("=" * 60)
