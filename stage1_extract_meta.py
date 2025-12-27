# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/24 20:59
@File    : stage1_extract_meta.py
@Author  : zj
@Description:

Stage 1: 深度解析每篇博客，生成结构化元数据（保存至 .tmp/）
✅ 支持缓存跳过
✅ 记录每篇耗时 & tokens
✅ 全文参与 LLM 分析（分块后汇总）
✅ 健壮的 API 调用与 JSON 输出（防 YAML 冒号问题）
✅ 自动修复轻微 JSON 截断
✅ LLM 自我校验（self-refine）确保格式规范（仅格式，不改内容）
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import frontmatter
import openai
import time
import sys
import json

# === 配置 ===
BLOG_POSTS_DIR = Path("blog/source/_posts")
TMP_OUTPUT_DIR = Path(".tmp")
TMP_OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "deepseek-chat"
openai.base_url = "https://api.deepseek.com/v1/"
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-")

MAX_RETRIES = 3
CHUNK_SIZE = 8000
OVERLAP = 500

global_stats = {
    'total_articles': 0,
    'processed_articles': 0,
    'skipped_articles': 0,
    'total_duration_sec': 0.0,
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
}


def clean_markdown(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    md = re.sub(r"`[^`]*`", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"^#+\s*", "", md, flags=re.MULTILINE)
    md = re.sub(r"^>.*$", "", md, flags=re.MULTILINE)
    ref_patterns = [
        r"(?i)##?\s*(参考|相关阅读|延伸阅读|推荐阅读|更多资料|参考资料|reference|further reading|致谢|鸣谢)"
    ]
    for pat in ref_patterns:
        parts = re.split(pat, md, maxsplit=1)
        md = parts[0]
    return md.strip()


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> Tuple[str, Dict[str, int], float]:
    start_time = time.time()
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=90
        )
        duration = time.time() - start_time
        content = response.choices[0].message.content.strip()
        usage = getattr(response, 'usage', None)
        usage_dict = {
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(usage, 'completion_tokens', 0),
            'total_tokens': getattr(usage, 'total_tokens', 0)
        }

        global_stats['total_prompt_tokens'] += usage_dict['prompt_tokens']
        global_stats['total_completion_tokens'] += usage_dict['completion_tokens']
        global_stats['total_tokens'] += usage_dict['total_tokens']

        return content, usage_dict, duration

    except Exception as e:
        duration = time.time() - start_time
        raise RuntimeError(f"API 调用失败 (耗时 {duration:.2f}s): {e}")


def robust_llm_call(prompt: str, max_retries=MAX_RETRIES, temperature=0.2, max_tokens=4096) -> Tuple[str, Dict, float]:
    for i in range(1, max_retries + 1):
        try:
            content, usage, duration = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
            out = content.strip()

            # 移除可能的 markdown 包裹
            if out.startswith("```json"):
                out = out[7:]
            elif out.startswith("```"):
                out = out[3:]
            if out.endswith("```"):
                out = out[:-3]
            out = out.strip()

            return out, usage, duration
        except Exception as e:
            wait = min(2 ** i, 10)
            print(f"  ⚠️  Retry {i}/{max_retries}: {e}, wait {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


def refine_meta_output(draft_json_str: str) -> str:
    """
    使用 LLM 仅修复 JSON 格式问题，严禁修改任何语义内容。
    """
    refine_prompt = f"""你是一位 JSON 格式校验器。请严格按以下规则处理输入：

【任务】
- 如果输入已是合法 JSON，请原样返回（不要改动任何一个字，包括标点、空格、措辞）。
- 如果输入存在格式问题（如缺少双引号、缺少逗号、未闭合括号、包含 Markdown 代码块等），请仅修复格式，使其成为合法 JSON。
- **绝对禁止修改 main_idea、tags 的实际文字内容**，即使内容有事实错误、逻辑不符或不符合规范，也必须原样保留。

【输出要求】
- 仅输出修复后的 JSON 字符串；
- 不要任何解释、注释、前缀、后缀；
- 必须使用双引号；
- 确保可被 Python json.loads() 解析。

【待处理内容】
{draft_json_str}

现在请输出修复后的 JSON："""

    try:
        refined, _, _ = robust_llm_call(refine_prompt, temperature=0.0, max_tokens=1024)
        return refined
    except Exception as e:
        print(f"  ⚠️ 校验失败，回退原始输出: {e}", file=sys.stderr)
        return draft_json_str


def build_single_chunk_prompt(title: str, content: str, existing_cats, existing_tags) -> str:
    ex_cat_str = ", ".join(flatten_list(existing_cats)) if existing_cats else "无"
    ex_tag_str = ", ".join(flatten_list(existing_tags)) if existing_tags else "无"
    # 替换原来的 build_single_chunk_prompt 中的 JSON 格式说明
    return f"""你是一位资深技术博客编辑，请深度分析以下文章，输出其结构化元信息。

    要求：
    1. **文章整体思想**（main_idea）：
    - 用 1~2 句话概括（总字数 ≤ 200 字）；
    - 必须体现文章性质，例如：
        - “本文是对《DVC: An End-to-end...》的详细解读。”
        - “这是一份 OpenDVC 开源项目的实现报告。”
        - “本文总结了作者 2024 年在 AI 工程化方向的学习与项目实践。”
    - 禁止使用英文单引号 '...'，论文标题请用中文书名号《...》
    2. **标签列表**（tags，最多 20 个）：
    - 格式：`中文/English`（如：光流/Optical Flow）；
    - 纯英文术语可直接写（如：YOLOv8）；
    - 必须是文章主动讲解或使用的技术实体。

    已有 front-matter（仅作参考）：
    - 分类: {ex_cat_str}
    - 标签: {ex_tag_str}

    文章标题: {title}
    文章正文:
    {content[:12000]}

    请**仅输出标准 JSON**，格式如下：
    {{"main_idea": "...", "tags": [...]}}

    ⚠️ 重要：
    - 不要任何解释、注释、markdown 代码块；
    - 字符串必须用双引号；
    - 不要包含任何英文单引号 '...'；
    - 论文标题请用《...》。
    """


def build_multi_chunk_final_prompt(title: str, combined_summary: str, existing_cats, existing_tags) -> str:
    ex_cat_str = ", ".join(flatten_list(existing_cats)) if existing_cats else "无"
    ex_tag_str = ", ".join(flatten_list(existing_tags)) if existing_tags else "无"
    return f"""基于以下摘要生成最终元信息。

要求：
- main_idea：1~2 句（≤200 字），体现文章性质，用《》标注论文名；
- tags：最多 20 个，格式 `中文/English`。

已有 front-matter:
- 分类: {ex_cat_str}
- 标签: {ex_tag_str}

文章标题: {title}
文章技术摘要:
{combined_summary}

请**仅输出标准 JSON**：
{{"main_idea": "...", "tags": [...]}}

⚠️ 不要任何额外文字！"""


def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(str(item).strip())
    return result


def try_fix_json(json_str: str) -> str:
    """尝试修复常见的 JSON 截断问题"""
    s = json_str.strip()
    if not s.endswith('}'):
        if '"tags": [' in s and not s.rstrip().endswith(']'):
            s = s.rstrip() + "]}"
        elif s.count('{') > s.count('}'):
            s = s.rstrip() + "}"
    return s


def extract_meta_from_article(file_path: Path) -> Tuple[Dict[str, Any], float, Dict]:
    start_time = time.time()
    out_file = TMP_OUTPUT_DIR / (file_path.stem + ".json")

    if out_file.exists():
        with open(out_file, 'r', encoding='utf-8') as f:
            try:
                cached = json.load(f)
                if cached and isinstance(cached, dict) and 'main_idea' in cached:
                    duration = time.time() - start_time
                    print(f"  💾 命中缓存，跳过解析", file=sys.stderr)
                    return cached, duration, {}
            except Exception as e:
                print(f"  ⚠️ 缓存损坏，重新解析: {e}", file=sys.stderr)

    post = frontmatter.load(str(file_path))
    title = post.get("title", file_path.stem)
    content = post.content
    existing_cats = post.get("categories", [])
    existing_tags = post.get("tags", [])

    clean_content = clean_markdown(content)
    chunks = split_text_into_chunks(clean_content)

    raw_output = ""
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    total_duration = 0.0

    if len(chunks) == 1:
        full_prompt = build_single_chunk_prompt(title, chunks[0], existing_cats, existing_tags)
        raw_output, usage, duration = robust_llm_call(full_prompt, max_tokens=1024)
        total_usage = usage
        total_duration = duration
    else:
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = f"""你是技术文档分析专家。请总结以下文章片段的核心技术内容（忽略示例、引用、链接）：
片段 {i + 1}/{len(chunks)}:
{chunk[:4000]}
---
仅输出一段简洁的技术摘要（50字内），不要编号。"""
            summary, usage, duration = robust_llm_call(prompt, temperature=0.1, max_tokens=256)
            summaries.append(summary)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            total_duration += duration

        combined_summary = " ".join(summaries)
        full_prompt = build_multi_chunk_final_prompt(title, combined_summary, existing_cats, existing_tags)
        raw_output, usage, duration = robust_llm_call(full_prompt, max_tokens=1024)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)
        total_duration += duration

    # === Step 1: 自我校验（仅格式）===
    refined_output = refine_meta_output(raw_output)

    # === 新增：如果 refine 输出为空或明显无效，回退到 raw_output ===
    def is_plausibly_json(s: str) -> bool:
        s = s.strip()
        return s.startswith('{') and s.endswith('}')

    final_output_to_parse = refined_output
    if not is_plausibly_json(refined_output):
        print("  ⚠️ refined 输出无效，尝试使用原始输出", file=sys.stderr)
        final_output_to_parse = raw_output

    # === Step 2: 尝试解析 ===
    data = None
    last_error = None
    for candidate in [final_output_to_parse, refined_output, raw_output]:
        try:
            fixed = try_fix_json(candidate.strip())
            data = json.loads(fixed)
            break
        except json.JSONDecodeError as e:
            last_error = e
            continue

    if data is None:
        print(f"  ❌ 所有尝试均失败: {last_error}", file=sys.stderr)
        preview_raw = raw_output[:500].replace('\n', '\\n')
        preview_refined = refined_output[:500].replace('\n', '\\n')
        print(f"  🔍 raw 前 500 字符: {preview_raw}", file=sys.stderr)
        print(f"  🔍 refined 前 500 字符: '{preview_refined}'", file=sys.stderr)

        # 保存调试文件
        (TMP_OUTPUT_DIR / (file_path.stem + ".raw")).write_text(raw_output, encoding="utf-8")
        (TMP_OUTPUT_DIR / (file_path.stem + ".refined")).write_text(refined_output, encoding="utf-8")

        return {
            "error": f"Failed to parse JSON after all attempts. Last error: {str(last_error)}"}, time.time() - start_time, {}

    if not isinstance(data, dict):
        raise ValueError("LLM 返回非字典结构")

    # 清洗 tags
    tags = data.get("tags", [])
    cleaned_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip()
        if not tag or re.fullmatch(r"\d{4}", tag):
            continue
        if "/" in tag:
            parts = [p.strip() for p in tag.split("/", 1)]
            if len(parts) == 2:
                zh_part, en_part = parts
                if zh_part.lower() == en_part.lower():
                    cleaned_tags.append(en_part)
                else:
                    cleaned_tags.append(f"{zh_part}/{en_part}")
            else:
                cleaned_tags.append(tag)
        else:
            cleaned_tags.append(tag)

    seen = set()
    deduped = []
    for t in cleaned_tags:
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    data["tags"] = deduped[:20]
    data["source_file"] = file_path.name
    return data, time.time() - start_time, total_usage


def main():
    md_files = list(BLOG_POSTS_DIR.rglob("*.md"))
    global_stats['total_articles'] = len(md_files)
    print(f"🔍 找到 {len(md_files)} 篇文章，开始深度解析...", file=sys.stderr)

    for i, fp in enumerate(md_files, 1):
        out_file = TMP_OUTPUT_DIR / (fp.stem + ".json")  # ← 改为 .json
        if out_file.exists():
            global_stats['skipped_articles'] += 1
            print(f"[{i}/{len(md_files)}] ⏭️  跳过（已存在）: {fp.name}")
            continue

        print(f"[{i}/{len(md_files)}] 🧠 解析: {fp.name}")
        try:
            meta, duration, usage = extract_meta_from_article(fp)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)  # ← 保存为标准 JSON
            global_stats['processed_articles'] += 1
            global_stats['total_duration_sec'] += duration

            pt = usage.get('prompt_tokens', 0)
            ct = usage.get('completion_tokens', 0)
            tt = usage.get('total_tokens', 0)
            print(f"  ✅ 耗时: {duration:.2f}s | Tokens: {pt}/{ct} → {tt} | 保存至 {out_file.name}")
        except Exception as e:
            print(f"  ❌ 跳过 {fp.name}: {e}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"🎉 Stage 1 完成！", file=sys.stderr)
    print(f"📊 总计: {global_stats['total_articles']} 篇", file=sys.stderr)
    print(f"   ✅ 处理: {global_stats['processed_articles']}", file=sys.stderr)
    print(f"   ⏭️  跳过: {global_stats['skipped_articles']}", file=sys.stderr)
    print(f"   ⏱️  总耗时: {global_stats['total_duration_sec']:.2f} 秒", file=sys.stderr)
    print(
        f"   🔢 总 Tokens: {global_stats['total_prompt_tokens']} + {global_stats['total_completion_tokens']} = {global_stats['total_tokens']}",
        file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
