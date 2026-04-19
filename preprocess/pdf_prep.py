#!/usr/bin/env python3
"""
PDF 批量预处理脚本
==================
功能：
  1. 批量处理论文 PDF，提取文本和图表
  2. 自动识别论文元数据（标题、作者、年份）
  3. 将图表保存为单独图片（用于视觉分析）
  4. 生成摄入清单（供后续 paper_processor.py 使用）
  5. 从文献管理软件（Zotero/Mendeley）导出的 BibTeX 补充元数据

依赖：
  pip install pypdf pdfplumber
  # 图表提取需要: pip install pdf2image pillow
  # pdf2image 还需要: sudo apt-get install poppler-utils（Linux）
  #                    brew install poppler（Mac）

用法：
  python preprocess/pdf_prep.py --input papers/ --output_dir data/papers/
  python preprocess/pdf_prep.py --input paper.pdf --bibtex refs.bib
  python preprocess/pdf_prep.py --input papers/ --extract_figures
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def extract_text_pypdf(pdf_path: str) -> str:
    """Extract text using pypdf (basic, fast)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n".join(texts)
    except Exception as e:
        return f"[提取失败: {e}]"


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (better for multi-column layouts)."""
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return "\n".join(texts)
    except ImportError:
        return extract_text_pypdf(pdf_path)
    except Exception as e:
        return extract_text_pypdf(pdf_path)


def guess_metadata_from_text(text: str, filename: str) -> dict:
    """
    Heuristically extract title, authors, year from PDF text.
    Works best for papers with standard formatting.
    """
    metadata = {
        "title": "",
        "authors": [],
        "year": None,
        "journal": "",
        "doi": "",
        "source_file": os.path.basename(filename),
    }

    lines = [l.strip() for l in text[:3000].split("\n") if l.strip()]

    # Year: look for 4-digit year pattern (1990-2030)
    year_match = re.search(r"\b(19[9]\d|20[012]\d)\b", text[:2000])
    if year_match:
        metadata["year"] = int(year_match.group(1))

    # DOI
    doi_match = re.search(r"10\.\d{4,}/[^\s]+", text[:3000])
    if doi_match:
        metadata["doi"] = doi_match.group(0).rstrip(".,)")

    # Title heuristic: usually first long line in the document
    for line in lines[:15]:
        if len(line) > 20 and not re.match(r"^\d", line):
            if not re.search(r"copyright|journal|received|published", line, re.I):
                metadata["title"] = line
                break

    # Authors: look for patterns like "Author1, Author2, Author3*"
    author_pattern = re.compile(
        r"([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)"
    )
    author_match = author_pattern.search(text[:2000])
    if author_match:
        raw = author_match.group(1)
        metadata["authors"] = [a.strip() for a in raw.split(",")][:6]

    # Try to improve title from filename
    if not metadata["title"] and filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        # Clean common filename patterns like "2024_Chen_SAC"
        clean = re.sub(r"_+", " ", base)
        clean = re.sub(r"^\d{4}\s*", "", clean)
        metadata["title"] = clean

    return metadata


def load_bibtex_metadata(bib_path: str) -> dict[str, dict]:
    """
    Parse a BibTeX file and return a dict keyed by title fragment.
    Enables matching PDF files to their metadata.
    """
    entries = {}
    try:
        with open(bib_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  ⚠️  BibTeX 文件不存在: {bib_path}")
        return entries

    # Simple regex-based BibTeX parser
    entry_pattern = re.compile(
        r"@\w+\{([^,]+),([^@]+)", re.DOTALL
    )
    for match in entry_pattern.finditer(content):
        key = match.group(1).strip()
        body = match.group(2)

        def get_field(name):
            m = re.search(
                rf"{name}\s*=\s*[\{{\"](.*?)[\}}\"]",
                body, re.DOTALL | re.IGNORECASE
            )
            return m.group(1).strip() if m else ""

        title = get_field("title")
        author = get_field("author")
        year = get_field("year")
        journal = get_field("journal") or get_field("booktitle")

        entries[key] = {
            "title": title,
            "authors": [a.strip() for a in author.split(" and ")] if author else [],
            "year": int(year) if year and year.isdigit() else None,
            "journal": journal,
        }

    print(f"  从 BibTeX 加载了 {len(entries)} 条文献记录")
    return entries


def match_metadata_from_bibtex(title_guess: str,
                                bib_entries: dict) -> dict | None:
    """Find the best matching BibTeX entry for a guessed title."""
    if not bib_entries or not title_guess:
        return None

    title_lower = title_guess.lower()
    best_score = 0
    best_entry = None

    for entry in bib_entries.values():
        bib_title = entry.get("title", "").lower()
        # Simple word overlap score
        title_words = set(title_lower.split())
        bib_words = set(bib_title.split())
        if not bib_words:
            continue
        overlap = len(title_words & bib_words) / max(len(bib_words), 1)
        if overlap > best_score and overlap > 0.4:
            best_score = overlap
            best_entry = entry

    return best_entry


def extract_figures(pdf_path: str, output_dir: str,
                    max_pages: int = 20) -> list[str]:
    """
    Convert PDF pages to images for figure analysis.
    Returns list of saved image paths.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("  图表提取需要: pip install pdf2image")
        print("  以及: sudo apt-get install poppler-utils")
        return []

    try:
        pages = convert_from_path(
            pdf_path, dpi=150, fmt="png",
            first_page=1, last_page=min(max_pages, 50),
        )
    except Exception as e:
        print(f"  图表提取失败: {e}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    saved = []
    for i, page in enumerate(pages):
        path = os.path.join(output_dir, f"{base}_page{i+1:02d}.png")
        page.save(path, "PNG")
        saved.append(path)

    print(f"  保存了 {len(saved)} 页图像")
    return saved


def process_single_pdf(pdf_path: str,
                        output_dir: str,
                        bib_entries: dict | None = None,
                        extract_figs: bool = False,
                        field_hint: list[str] | None = None) -> dict:
    """Process a single PDF file."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n处理: {os.path.basename(pdf_path)}")

    # Extract text
    text = extract_text_pdfplumber(pdf_path)
    word_count = len(text.split())
    print(f"  提取文本: {word_count} 词")

    if word_count < 100:
        print("  ⚠️  文本量太少，可能是扫描版 PDF（需要 OCR）")

    # Extract/guess metadata
    meta = guess_metadata_from_text(text, pdf_path)

    # Try to improve with BibTeX
    if bib_entries and meta["title"]:
        bib_match = match_metadata_from_bibtex(meta["title"], bib_entries)
        if bib_match:
            # BibTeX data is more reliable
            if bib_match.get("title"):
                meta["title"] = bib_match["title"]
            if bib_match.get("authors"):
                meta["authors"] = bib_match["authors"]
            if bib_match.get("year"):
                meta["year"] = bib_match["year"]
            if bib_match.get("journal"):
                meta["journal"] = bib_match["journal"]
            print(f"  ✓ 匹配到 BibTeX 记录: {meta['title'][:50]}")

    # Add field hint
    if field_hint:
        meta["field"] = field_hint

    # Save extracted text
    text_dir = os.path.join(output_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)
    text_path = os.path.join(text_dir, f"{base}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text[:50000])  # Limit to first 50K chars

    # Extract figures
    figure_paths = []
    if extract_figs:
        figs_dir = os.path.join(output_dir, "figures", base)
        figure_paths = extract_figures(pdf_path, figs_dir)

    result = {
        "pdf_path": pdf_path,
        "text_path": text_path,
        "metadata": meta,
        "figure_paths": figure_paths,
        "word_count": word_count,
        "ready_for_ingestion": word_count >= 100,
    }

    print(f"  标题: {meta['title'][:60] or '(未识别)'}")
    print(f"  作者: {', '.join(meta['authors'][:3]) or '(未识别)'}")
    print(f"  年份: {meta['year'] or '(未识别)'}")

    return result


def generate_ingestion_manifest(results: list[dict],
                                 output_path: str,
                                 speaker_for_impressions: str = ""):
    """
    Generate a manifest file listing papers ready for ingestion.
    This file is read by the batch ingestion script.
    """
    manifest = {
        "target_researcher": speaker_for_impressions,
        "total_papers": len(results),
        "ready_count": sum(1 for r in results if r["ready_for_ingestion"]),
        "papers": [],
    }

    for r in results:
        meta = r["metadata"]
        manifest["papers"].append({
            "pdf_path": r["pdf_path"],
            "text_path": r["text_path"],
            "figure_paths": r.get("figure_paths", []),
            "metadata": {
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "year": meta.get("year"),
                "journal": meta.get("journal", ""),
                "field": meta.get("field", []),
                "doi": meta.get("doi", ""),
            },
            "word_count": r["word_count"],
            "ready": r["ready_for_ingestion"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 摄入清单已生成: {output_path}")
    print(f"   可用: {manifest['ready_count']}/{manifest['total_papers']} 篇")
    print(f"\n下一步运行批量摄入:")
    print(f"   python preprocess/batch_ingest.py --manifest {output_path}")


def batch_process(input_dir: str, output_dir: str,
                  bib_path: str | None = None,
                  extract_figs: bool = False,
                  field_hint: list[str] | None = None,
                  speaker: str = "") -> list[dict]:
    """Process all PDFs in a directory."""
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"在 {input_dir} 中没有找到 PDF 文件")
        return []

    print(f"找到 {len(pdf_files)} 个 PDF 文件")

    # Load BibTeX if provided
    bib_entries = {}
    if bib_path:
        bib_entries = load_bibtex_metadata(bib_path)

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for pdf in pdf_files:
        try:
            result = process_single_pdf(
                str(pdf), output_dir, bib_entries, extract_figs, field_hint
            )
            results.append(result)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")

    # Generate manifest
    manifest_path = os.path.join(output_dir, "ingestion_manifest.json")
    generate_ingestion_manifest(results, manifest_path, speaker)

    return results


def main():
    parser = argparse.ArgumentParser(description="PDF 批量预处理工具")
    parser.add_argument("--input", required=True,
                        help="输入 PDF 文件或目录")
    parser.add_argument("--output_dir", default="data/papers_processed",
                        help="输出目录")
    parser.add_argument("--bibtex", default=None,
                        help="BibTeX 文件路径（用于补充元数据）")
    parser.add_argument("--extract_figures", action="store_true",
                        help="同时提取图表图像（需要 pdf2image 和 poppler）")
    parser.add_argument("--field", nargs="+", default=None,
                        help='领域标签，如 --field 氢能 催化剂')
    parser.add_argument("--speaker", default="",
                        help="目标科研人员姓名（写入摄入清单）")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_process(
            args.input, args.output_dir,
            args.bibtex, args.extract_figures,
            args.field, args.speaker,
        )
    else:
        bib_entries = {}
        if args.bibtex:
            bib_entries = load_bibtex_metadata(args.bibtex)
        process_single_pdf(
            args.input, args.output_dir,
            bib_entries, args.extract_figures, args.field,
        )


if __name__ == "__main__":
    main()
