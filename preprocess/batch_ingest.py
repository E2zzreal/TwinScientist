#!/usr/bin/env python3
"""
批量摄入脚本
============
读取 pdf_prep.py 生成的 ingestion_manifest.json，
批量调用 paper_processor.py 生成论文认知印象，
并自动更新 memory/topic_index.yaml。

用法：
  # 先运行预处理
  python preprocess/pdf_prep.py --input papers/ --output_dir data/papers_processed/

  # 再批量摄入
  python preprocess/batch_ingest.py --manifest data/papers_processed/ingestion_manifest.json

  # 指定领域（自动分类到话题索引）
  python preprocess/batch_ingest.py --manifest data/papers_processed/ingestion_manifest.json \\
      --topic hydrogen_catalyst
"""

import argparse
import json
import os
import sys
import time
import yaml


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_topic_index(memory_dir: str, topic_id: str,
                        paper_filename: str, paper_meta: dict):
    """Add a paper to the topic index."""
    index_path = os.path.join(memory_dir, "topic_index.yaml")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f) or {"topics": {}}
    else:
        index = {"topics": {}}

    topics = index.setdefault("topics", {})

    if topic_id not in topics:
        topics[topic_id] = {
            "summary": f"待完善 — 包含 {paper_meta.get('title', '')[:30]}等论文",
            "paper_count": 0,
            "stance": "待通过对话确认立场",
            "detail_files": [],
        }

    topic = topics[topic_id]
    rel_path = f"papers/{paper_filename}"
    if rel_path not in topic.get("detail_files", []):
        topic.setdefault("detail_files", []).append(rel_path)
        topic["paper_count"] = len(topic["detail_files"])

    with open(index_path, "w", encoding="utf-8") as f:
        yaml.dump(index, f, allow_unicode=True, default_flow_style=False)


def ingest_paper(agent, paper_info: dict, memory_dir: str,
                 topic_id: str | None = None) -> dict:
    """Ingest a single paper from the manifest."""
    from ingestion.paper_processor import process_paper

    meta = paper_info["metadata"]
    pdf_path = paper_info["pdf_path"]

    # Use pre-extracted text if available (faster, avoids re-reading PDF)
    text_path = paper_info.get("text_path")
    if text_path and os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            pre_extracted_text = f.read()
    else:
        pre_extracted_text = None

    try:
        # Call paper_processor (it will re-extract text from PDF if no pre-extracted)
        output_path = process_paper(
            client=agent.client,
            model=agent.config["model"],
            pdf_path=pdf_path,
            memory_dir=memory_dir,
            metadata={
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "year": meta.get("year"),
                "field": meta.get("field", []),
                "doi": meta.get("doi", ""),
            },
        )
        paper_filename = os.path.basename(output_path)

        # Update topic index
        if topic_id:
            update_topic_index(memory_dir, topic_id, paper_filename, meta)

        return {"status": "success", "output": output_path}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_batch_ingest(manifest_path: str,
                     project_dir: str = ".",
                     topic_id: str | None = None,
                     skip_existing: bool = True,
                     delay_seconds: float = 1.0):
    """Main batch ingestion runner."""
    from agent.main import TwinScientist

    manifest = load_manifest(manifest_path)
    papers = [p for p in manifest["papers"] if p.get("ready", True)]
    total = len(papers)

    print(f"批量摄入: {total} 篇论文")
    print(f"话题ID: {topic_id or '(不自动分类)'}")
    print(f"跳过已存在: {skip_existing}")
    print()

    agent = TwinScientist(project_dir)
    memory_dir = agent.memory_dir

    results = {"success": [], "skip": [], "error": []}

    for i, paper in enumerate(papers, 1):
        title = paper["metadata"].get("title", "未知")[:50]
        print(f"[{i}/{total}] {title}")

        # Check if already ingested
        if skip_existing:
            papers_dir = os.path.join(memory_dir, "papers")
            if os.path.exists(papers_dir):
                # Heuristic: check if a file with similar name exists
                year = paper["metadata"].get("year", "")
                existing = os.listdir(papers_dir)
                if any(str(year) in f for f in existing if title[:10].lower() in f.lower()):
                    print("  → 跳过（已存在）")
                    results["skip"].append(paper)
                    continue

        result = ingest_paper(agent, paper, memory_dir, topic_id)

        if result["status"] == "success":
            print(f"  ✅ 已保存: {os.path.basename(result['output'])}")
            results["success"].append(paper)
        else:
            print(f"  ❌ 失败: {result['error']}")
            results["error"].append({**paper, "error": result["error"]})

        # Rate limiting (avoid overwhelming the API)
        if i < total:
            time.sleep(delay_seconds)

    print(f"\n{'='*50}")
    print(f"完成: {len(results['success'])} 成功 / "
          f"{len(results['skip'])} 跳过 / "
          f"{len(results['error'])} 失败")

    if results["error"]:
        print("\n失败的论文:")
        for p in results["error"]:
            print(f"  - {p['metadata'].get('title', '?')}: {p.get('error', '')}")

    # Save ingestion log
    log_dir = os.path.dirname(manifest_path)
    log_path = os.path.join(log_dir, "ingestion_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\n摄入日志: {log_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="批量论文摄入工具")
    parser.add_argument("--manifest", required=True,
                        help="pdf_prep.py 生成的 ingestion_manifest.json")
    parser.add_argument("--project_dir", default=".",
                        help="TwinScientist 项目目录（默认: 当前目录）")
    parser.add_argument("--topic", default=None,
                        help="话题ID（如 hydrogen_catalyst），自动更新 topic_index")
    parser.add_argument("--no_skip", action="store_true",
                        help="不跳过已存在的论文（重新摄入）")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每篇论文之间的延迟秒数（默认: 1.0）")
    args = parser.parse_args()

    run_batch_ingest(
        manifest_path=args.manifest,
        project_dir=args.project_dir,
        topic_id=args.topic,
        skip_existing=not args.no_skip,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
