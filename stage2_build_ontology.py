# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/24 21:02
@File    : stage2_build_ontology.py
@Author  : zj
@Description:

Stage 2: 构建统一标签 & 分类体系（支持每篇文章最多 3 个 [主类, 子类] 类别）

输入: .tmp/*.json （来自 stage1）
输出:
  - .ontology/tags.json        # 标准化标签 + 别名
  - .ontology/categories.json  # 全局分类体系（[主类, 子类] 列表）
  - .ontology/assignment.json  # 每篇文章的 file / categories / tags

类别说明:
  - 每个类别是一个二元列表: ["主类", "子类"]
  - 每篇文章最多分配 3 个这样的类别
  - 主类反映文章性质（如"论文解读"），子类反映领域或目标（如"计算机视觉"）

目标：
1. 标签标准化：合并同义标签，输出 tags.json
   - 每个标准标签形如 "中文/English"
   - 记录所有别名
2. 分类体系构建：基于 content_type + main_idea，输出 categories.json
   - 聚焦“写作目的”而非技术主题
3. 为每篇文章分配最终 categories 和 tags，输出 assignment.json

注意：不吝惜 tokens，多次调用 LLM，确保质量。
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import openai
import time
import sys

# === 配置 ===
TMP_DIR = Path(".tmp")
OUTPUT_DIR = Path(".ontology")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "deepseek-chat"
openai.base_url = "https://api.deepseek.com/v1/"
openai.api_key = "sk-"

# === 全局统计 ===
STATS = {
    'total_calls': 0,
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
}


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """调用 LLM 并更新统计"""
    for attempt in range(3):
        try:
            start = time.time()
            resp = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120
            )
            duration = time.time() - start
            content = resp.choices[0].message.content.strip()

            usage = getattr(resp, 'usage', None)
            pt = getattr(usage, 'prompt_tokens', 0)
            ct = getattr(usage, 'completion_tokens', 0)
            STATS['total_calls'] += 1
            STATS['total_prompt_tokens'] += pt
            STATS['total_completion_tokens'] += ct

            print(f"  ✅ LLM 成功 | 耗时: {duration:.1f}s | Tokens: {pt}/{ct}")
            return content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠️ 重试 {attempt + 1}/3: {e}, 等待 {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM 调用失败")


def load_all_meta() -> Dict[str, Dict]:
    """加载所有 .tmp/*.json 文件"""
    metas = {}
    for f in TMP_DIR.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if isinstance(data, dict) and 'error' not in data:
                    metas[f.stem] = data
        except Exception as e:
            print(f"⚠️ 跳过损坏文件 {f.name}: {e}", file=sys.stderr)
    print(f"✅ 加载 {len(metas)} 篇有效文章元数据")
    return metas


# ======================
# 第一步：标签标准化
# ======================

def extract_tag_parts_fallback(tag: str) -> Tuple[str, str]:
    """
    仅做格式解析，不做语义归一。
    用于 fallback 构造 standard_tag。
    """
    tag = tag.strip()
    if '/' in tag:
        parts = [p.strip() for p in tag.split('/', 1)]
        zh = parts[0] or parts[1]
        en = parts[1] or parts[0]
    else:
        # 如果包含中文字符，视为中文
        if any('\u4e00' <= c <= '\u9fff' for c in tag):
            zh, en = tag, ""
        else:
            zh, en = "", tag
    return zh, en


def normalize_tags(all_raw_tags: List[str]) -> List[Dict[str, Any]]:
    """
    高精度标签标准化流程（三阶段）：

    1. 【语义聚类】将原始标签按语义分组（每批 ≤30 个，调用 LLM 聚类）
    2. 【簇标准化】对每个语义簇选出 standard_tag 并记录 aliases
    3. 【全局去重】合并跨簇冲突（如 A→B 和 B→C 合并为 A,B,C→统一标准）

    输入：所有原始标签（可能含重复、空格、大小写变体）
    输出：[{"standard_tag": "中文/English", "aliases": [...]}]

    注意：不依赖预定义词典，完全由 LLM 驱动语义理解。
    """
    unique_tags = sorted({t.strip() for t in all_raw_tags if t.strip()})
    if not unique_tags:
        return []

    print(f"🏷️ 共 {len(unique_tags)} 个唯一原始标签，开始纯语义标准化...")

    # ==============================
    # 阶段 1: 分批聚类（每批最多 30 个标签）
    # ==============================
    all_clusters: List[List[str]] = []
    batch_size = 30

    # 计算总批次数，用于进度显示
    total_batches = (len(unique_tags) + batch_size - 1) // batch_size

    for i in range(0, len(unique_tags), batch_size):
        batch_index = i // batch_size + 1  # 从 1 开始计数
        batch = unique_tags[i:i + batch_size]
        if len(batch) == 1:
            all_clusters.append(batch)
            print(f"[{batch_index}/{total_batches}] 跳过单标签批次: {batch[0]}")
            continue

        tags_str = "\n".join(f"- {tag}" for tag in batch)
        prompt = f"""你是一位极度严谨的技术术语标准化专家。你的任务是：**仅当两个或多个标签完全等价时才将它们归为一组**，否则每个标签必须单独成组。

        📌 合并的唯一合法情形（必须满足以下之一）：
        - 完全相同的术语，仅格式不同（如大小写、连字符、空格）：["ResNet50", "ResNet-50"]
        - 官方同义词或别名（如 "BERT" 和 "Bidirectional Encoder Representations from Transformers"）
        - 中英文对照且明确指代同一事物：["卷积神经网络", "CNN"]
        - 缩写与全称一一对应：["LLM", "Large Language Model"]

        🚫 以下情况**绝对禁止合并**（即使看起来相关）：
        - 不同算法（如 Adam ≠ AdaGrad ≠ RMSProp）
        - 不同数据集（如 CIFAR-10 ≠ ImageNet ≠ COCO）
        - 不同架构（如 AlexNet ≠ ResNet ≠ Transformer）
        - 不同概念但首字母相同（如 API ≠ AML ≠ AP@n）
        - 一个通用词 + 一个具体工具（如 "AI识别" ≠ "ARM G52"）
        - 任何你无法 100% 确定可互换的情况

        ⚠️ 重要指令：
        - 如果你对任意两个标签是否等价存在**丝毫犹豫，请将它们分开**
        - 宁可输出 30 个单元素组，也不要输出 1 个错误合并组
        - 每个标签必须出现在且仅出现在一个组中

        标签列表（共 {len(batch)} 个）：
        {tags_str}

        输出格式：严格为 JSON 列表，每组是一个字符串数组。
        ✅ 正确示例：
        [
          ["LLM", "Large Language Model"],
          ["ResNet50"],
          ["Adam"],
          ["AdaGrad"]
        ]
        ❌ 错误示例（不要这样做）：
        [
          ["Adam", "AdaGrad"],  // 不同优化器！
          ["API", "AML"]       // 完全无关！
        ]

        现在，请输出聚类结果（仅 JSON，无其他内容）：
        """

        print(
            f"[{batch_index}/{total_batches}] 🧠 LLM 聚类批次（{len(batch)} 个标签）: {batch[:3]}{'...' if len(batch) > 3 else ''}")
        try:
            output = call_llm(prompt, temperature=0.0, max_tokens=1024)
            if output.startswith("```json"):
                output = output[7:-3].strip()
            elif output.startswith("```"):
                output = output[3:-3].strip()
            clusters = json.loads(output)

            covered = set()
            valid_clusters = []
            for cluster in clusters:
                if not isinstance(cluster, list):
                    continue
                clean_cluster = []
                for tag in cluster:
                    if tag in batch and tag not in covered:
                        clean_cluster.append(tag)
                        covered.add(tag)
                if clean_cluster:
                    valid_clusters.append(clean_cluster)
            missing = [tag for tag in batch if tag not in covered]
            for tag in missing:
                valid_clusters.append([tag])
            all_clusters.extend(valid_clusters)
            print(f"  ✅ 批次 {batch_index} 聚类成功，生成 {len(valid_clusters)} 个簇")

            # === 新增：打印合并的语义簇（仅显示长度 ≥2 的）===
            merged_clusters = [c for c in valid_clusters if len(c) > 1]
            if merged_clusters:
                print(f"    🔗 合并的同义标签组（共 {len(merged_clusters)} 组）:")
                for cid, cluster in enumerate(merged_clusters, 1):
                    print(f"      {cid}. {cluster}")
            else:
                print("    ➖ 无合并，所有标签独立成簇")
        except Exception as e:
            print(f"⚠️ 批次聚类失败（{len(batch)} 个标签），回退为单标签组: {e}")
            for tag in batch:
                all_clusters.append([tag])

    print(f"✅ 聚类完成，得到 {len(all_clusters)} 个语义簇")

    print(f"📊 簇大小分布:")
    size_count = defaultdict(int)
    for c in all_clusters:
        size_count[len(c)] += 1
    for size in sorted(size_count):
        print(f"  长度 {size}: {size_count[size]} 个簇")

    # ==============================
    # 阶段 2: 对每个簇标准化
    # ==============================
    standardized = []

    for idx, cluster in enumerate(all_clusters, 1):
        print(f"[{idx}/{len(all_clusters)}] 标准化簇: {cluster[:3]}{'...' if len(cluster) > 3 else ''}")
        cluster = list(dict.fromkeys(cluster))
        if len(cluster) == 1:
            tag = cluster[0]
            zh, en = extract_tag_parts_fallback(tag)
            std_tag = f"{zh}/{en}" if zh and en else (zh or en or tag)
            standardized.append({
                "standard_tag": std_tag,
                "aliases": []
            })
            continue

        if len(cluster) > 20:
            for j in range(0, len(cluster), 20):
                sub = cluster[j:j + 20]
                _standardize_cluster(sub, standardized)
        else:
            _standardize_cluster(cluster, standardized)

    # ==============================
    # 阶段 3: 全局去重 & 冲突解决
    # ==============================
    std_to_aliases: Dict[str, Set[str]] = defaultdict(set)
    tag_to_std: Dict[str, str] = {}

    for item in standardized:
        std = item["standard_tag"]
        aliases = item["aliases"]
        alias_set = {a for a in aliases if a != std}
        std_to_aliases[std].update(alias_set)
        tag_to_std[std] = std
        for a in alias_set:
            if a in tag_to_std and tag_to_std[a] != std:
                old_std = tag_to_std[a]
                std_to_aliases[std].update(std_to_aliases[old_std])
                std_to_aliases[std].discard(std)
                for t in [old_std] + list(std_to_aliases[old_std]):
                    tag_to_std[t] = std
                std_to_aliases.pop(old_std, None)
            else:
                tag_to_std[a] = std

    final_result = []
    emitted_std = set()
    for std, aliases in std_to_aliases.items():
        if std in emitted_std:
            continue
        emitted_std.add(std)
        clean_aliases = [a for a in aliases if a not in std_to_aliases]
        final_result.append({
            "standard_tag": std,
            "aliases": clean_aliases
        })

    all_covered = set(tag_to_std.keys())
    missing = [t for t in unique_tags if t not in all_covered]
    for t in missing:
        zh, en = extract_tag_parts_fallback(t)
        std_tag = f"{zh}/{en}" if zh and en else (zh or en or t)
        final_result.append({
            "standard_tag": std_tag,
            "aliases": []
        })

    print(f"🎯 标签标准化完成：共 {len(final_result)} 个标准标签")
    return final_result


def _standardize_cluster(cluster: List[str], result_list: List[Dict[str, Any]]):
    tags_str = "\n".join(f"- {tag}" for tag in cluster)
    prompt = f"""以下是一组语义相同或高度相关的标签。请选择其中一个作为 standard_tag，
要求：
1. 优先选择包含中英文的形式（如“大模型/Large Language Model”）
2. 若无双语，选最完整、规范的形式
3. standard_tag 必须是以下列表中的一个

标签列表：
{tags_str}

输出严格为 JSON：
{{
  "standard_tag": "选中的标签",
  "aliases": ["其余标签"]
}}
注意：aliases 不得包含 standard_tag 本身。
"""

    try:
        output = call_llm(prompt, temperature=0.1, max_tokens=512)
        if output.startswith("```json"):
            output = output[7:-3].strip()
        elif output.startswith("```"):
            output = output[3:-3].strip()
        result = json.loads(output)
        std = result["standard_tag"]
        aliases = result.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        if std not in cluster:
            raise ValueError("standard_tag not in input")
        valid_aliases = [a for a in aliases if a in cluster and a != std]
        remaining = [t for t in cluster if t != std and t not in valid_aliases]
        valid_aliases.extend(remaining)

        result_list.append({
            "standard_tag": std,
            "aliases": valid_aliases
        })
    except Exception as e:
        print(f"  ⚠️ 簇标准化失败，回退选最长标签: {e}")
        std = max(cluster, key=len)
        aliases = [t for t in cluster if t != std]
        result_list.append({
            "standard_tag": std,
            "aliases": aliases
        })


# ======================
# 第二步：构建分类体系（输出 [主类, 子类] 列表）
# ======================

def build_category_system(article_metas: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    直接从 .ontology/category_schema.json 加载预定义分类体系，
    不再通过 LLM 动态生成子类。

    输出格式：
      - 不可细分主类 → ["主类"]
      - 可细分主类 → ["主类", "子类"]（使用 schema 中的 subcategories）
    """
    schema_path = Path(".ontology/category_schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到分类架构文件: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    result_categories = []

    for cat_def in schema["main_categories"]:
        main_name = cat_def["name"]
        description = cat_def["description"]
        allow_sub = cat_def.get("allow_subcategory", False)

        if not allow_sub:
            # 单层路径
            result_categories.append({
                "path": [main_name],
                "description": description
            })
        else:
            # 双层路径：使用预定义的 subcategories
            subcats = cat_def.get("subcategories", [])
            if not subcats:
                # 若无子类，至少保留一个占位（避免主类无子类）
                subcats = ["通用实践"]
            for sub in subcats:
                result_categories.append({
                    "path": [main_name, sub],
                    "description": f"{description} —— 具体方向：{sub}"
                })

    print(f"✅ 从 category_schema.json 加载 {len(result_categories)} 个预定义分类")
    for idx, cat in enumerate(result_categories, 1):
        path_str = " / ".join(cat["path"])
        print(f"  {idx}. {path_str} —— {cat['description']}")

    return result_categories


# ======================
# 第三步：分配（最多 3 个 [主类, 子类]）
# ======================


def assign_categories_and_tags(
        article_metas: Dict[str, Dict],
        categories: List[Dict],
        standardized_tags: List[Dict]
) -> List[Dict]:
    """
    为每篇文章分配分类路径和标准化标签。

    路径规则：
      - 不可细分主类（allow_subcategory=false）→ 仅允许单层路径，如 ["人生感悟"]
      - 可细分主类（allow_subcategory=true）→ 必须为双层路径，如 ["论文精读", "扩散模型"]
    """
    from pathlib import Path

    # === 1. 加载主类 schema（动态获取规则）===
    schema_path = Path(".ontology/category_schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到分类架构文件: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    main_defs = {cat["name"]: cat for cat in schema["main_categories"]}
    VALID_MAINS = set(main_defs.keys())
    NON_SUBDIVIDABLE = {name for name, d in main_defs.items() if not d["allow_subcategory"]}
    SUBDIVIDABLE = VALID_MAINS - NON_SUBDIVIDABLE

    # === 2. 构建标准标签映射（含别名）===
    alias_to_std = {}
    for item in standardized_tags:
        std_tag = item.get("standard_tag", "").strip()
        if not std_tag:
            continue
        alias_to_std[std_tag] = std_tag
        for alias in item.get("aliases", []):
            alias_clean = alias.strip()
            if alias_clean:
                alias_to_std[alias_clean] = std_tag

    # === 3. 构建严格合法的路径集合（tuple 形式便于查找）===
    valid_paths = set()
    for cat in categories:
        path = cat.get("path")
        if not isinstance(path, list) or not (1 <= len(path) <= 2):
            continue
        main = path[0]
        if main not in VALID_MAINS:
            continue
        # 强制层级合规
        if main in SUBDIVIDABLE and len(path) != 2:
            continue  # 可细分主类必须双层
        if main in NON_SUBDIVIDABLE and len(path) != 1:
            continue  # 不可细分主类必须单层
        valid_paths.add(tuple(path))

    if not valid_paths:
        print("⚠️ 无有效分类路径，跳过分配")
        return [{"file": f"{stem}.md", "categories": [], "tags": []} for stem in article_metas]

    # === 4. 开始分配 ===
    total = len(article_metas)
    print(f"🎯 开始分配分类与标签（共 {total} 篇文章）...")

    ASSIGN_PROMPT = """你是一位严谨的博客分类专家。请为以下文章从**下方明确列出的路径中**选择最相关的分类。

📌 要求：
- 最多选择 3 个分类路径；
- **必须严格使用“可用分类路径”中的条目**；
- 路径格式说明：
  • 单层：["人生感悟"] —— 仅用于不可细分主类
  • 双层：["踩坑记录", "CUDA配置"] —— 用于其余所有主类
- 如果没有相关项，返回空列表。

可用分类路径（共 {n} 条）：
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

    assignments = []
    failed = 0

    # 排序路径以便展示更清晰（先按主类，再按是否双层）
    display_paths = sorted(valid_paths, key=lambda p: (p[0], len(p), p[1] if len(p) > 1 else ""))
    cat_options = "\n".join(f"- {list(p)}" for p in display_paths)

    for idx, (stem, meta) in enumerate(article_metas.items(), 1):
        print(f"[{idx}/{total}] 📄 分配: {stem}")

        # --- 标准化标签 ---
        raw_tags = meta.get("tags", [])
        final_tags = []
        for rt in raw_tags:
            rt_clean = str(rt).strip()
            if rt_clean:
                final_tags.append(alias_to_std.get(rt_clean, rt_clean))

        # --- 准备 LLM 输入 ---
        idea = meta.get("main_idea", "").strip() or "（无主旨）"
        type_hint = meta.get("content_type", "").strip() or "（未知类型）"
        tags_str = ", ".join(final_tags[:10]) if final_tags else "（无标签）"

        final_cats = []
        try:
            prompt = ASSIGN_PROMPT.format(
                n=len(display_paths),
                cat_options=cat_options,
                type_hint=type_hint,
                idea=idea,
                tags_str=tags_str
            )
            output = call_llm(prompt, temperature=0.0, max_tokens=512)

            # 清理可能的 markdown 包裹
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
            failed += 1
            print(f"  ⚠️ 分配失败 ({stem}): {type(e).__name__}: {e}")
            final_cats = []

        assignments.append({
            "file": f"{stem}.md",
            "categories": final_cats,
            "tags": final_tags
        })

    print(f"✅ 分类与标签分配完成！共 {total} 篇，失败 {failed} 篇")
    return assignments


# ======================
# 主函数
# ======================

def main():
    print("📥 Stage 2: 构建统一 Ontology（支持最多 3 个 [主类, 子类] 类别）")

    metas = load_all_meta()
    if not metas:
        print("❌ 未找到有效元数据，请先运行 stage1")
        return

    # Step 1: 标准化标签（全局一次，支持缓存）
    print("\n🏷️ 步骤 1: 标准化标签...")

    tags_cache_path = OUTPUT_DIR / "tags.json"

    if tags_cache_path.exists():
        print(f"📂 检测到标签缓存文件: {tags_cache_path}")
        try:
            with open(tags_cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            standardized_tags = cache_data.get("tags", [])
            if standardized_tags:
                print(f"✅ 成功加载 {len(standardized_tags)} 个标准化标签（跳过 LLM 聚类）")
            else:
                print("⚠️ 缓存文件中无有效标签，将重新生成...")
                standardized_tags = None
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败 ({e})，将重新生成标签...")
            standardized_tags = None
    else:
        standardized_tags = None

    # 如果没有有效缓存，则执行标准化
    if standardized_tags is None:
        all_raw_tags = [tag for m in metas.values() for tag in m.get("tags", [])]
        standardized_tags = normalize_tags(all_raw_tags)
        # 保存缓存
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tags_cache_path, "w", encoding="utf-8") as f:
            json.dump({"tags": standardized_tags}, f, ensure_ascii=False, indent=2)
        print(f"💾 标准化标签已保存至: {tags_cache_path}")

    # Step 2: 构建分类体系（直接加载预定义 schema，无需缓存或 LLM）
    print("\n🗂️ 步骤 2: 加载预定义分类体系（来自 category_schema.json）...")

    categories = build_category_system(metas)
    categories_cache_path = OUTPUT_DIR / "categories.json"
    with open(categories_cache_path, "w", encoding="utf-8") as f:
        json.dump({"categories": categories}, f, ensure_ascii=False, indent=2)
    print(f"💾 分类体系已保存至: {categories_cache_path}")

    # Step 3: 分配（使用全局映射）
    print("\n🎯 步骤 3: 分配类别与标签（每篇最多 3 个 [主类, 子类]）...")
    assignments = assign_categories_and_tags(metas, categories, standardized_tags)
    with open(OUTPUT_DIR / "assignment.json", "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n🎉 Stage 2 完成！")
    print(f"  标准化标签数: {len(standardized_tags)}")
    print(f"  全局分类数: {len(categories)}")
    print(f"  文章分配数: {len(assignments)}")
    print(f"\n📊 LLM 统计:")
    print(f"  调用次数: {STATS['total_calls']}")
    print(f"  总 tokens: {STATS['total_prompt_tokens'] + STATS['total_completion_tokens']}")


if __name__ == "__main__":
    main()
