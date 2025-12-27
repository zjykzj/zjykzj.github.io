# -*- coding: utf-8 -*-

"""
@Time    : 2025/12/24 21:05
@File    : stage3_update_frontmatter.py
@Author  : zj
@Description: 

Stage 3: 根据 assignment.yaml 更新每篇博文的 front-matter
注意：assignment 中的 "file" 字段仅为文件名（如 "xxx.md"），需在 posts 目录中递归查找真实路径
"""

import yaml
from pathlib import Path
import frontmatter
import shutil

ASSIGNMENT_FILE = Path(".ontology/assignment.json")
BACKUP_DIR = Path(".backup_stage3")
POSTS_ROOT = Path("blog/source/_posts")  # 👈 请根据你的项目结构调整


def build_file_map(root_dir: Path) -> dict[str, Path]:
    """构建 {文件名: 完整路径} 的映射"""
    file_map = {}
    for md_file in root_dir.rglob("*.md"):
        filename = md_file.name
        if filename in file_map:
            print(f"⚠️ 警告：重复文件名 {filename}，路径 {file_map[filename]} 将被 {md_file} 覆盖")
        file_map[filename] = md_file
    return file_map


def main():
    BACKUP_DIR.mkdir(exist_ok=True)

    # 加载分配结果
    with open(ASSIGNMENT_FILE, "r", encoding="utf-8") as f:
        assignments = yaml.safe_load(f)

    # 构建文件名到路径的映射
    file_map = build_file_map(POSTS_ROOT)
    print(f"🔍 在 {POSTS_ROOT} 下找到 {len(file_map)} 篇 Markdown 文件")

    updated = 0
    not_found = 0
    for item in assignments:
        filename = item["file"]  # 例如 "PyTorch-Numpy-Softmax-计算概率.md"
        if filename not in file_map:
            print(f"❌ 未找到文件: {filename}")
            not_found += 1
            continue

        real_path = file_map[filename]
        print(f"🔄 匹配: {filename} → {real_path}")

        # 备份
        shutil.copy2(real_path, BACKUP_DIR / filename)

        # 更新 front-matter
        post = frontmatter.load(str(real_path))
        post["categories"] = item["categories"]
        post["tags"] = item["tags"]

        with open(real_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        updated += 1

    print(f"\n✅ 成功更新 {updated} 篇文章")
    if not_found:
        print(f"❌ 有 {not_found} 个文件未找到，请检查文件名是否一致")
    print(f"📁 备份已保存至: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
