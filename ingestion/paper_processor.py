# ingestion/paper_processor.py
"""Convert research papers (PDF) into structured cognitive impressions."""
import os
import re
import yaml
from datetime import datetime

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

IMPRESSION_PROMPT = """你正在帮助一位材料科学家（专攻氢能）建立论文认知档案。

以下是一篇论文的文本内容（可能被截断）：

---
{paper_text}
---

论文元数据：
标题：{title}
年份：{year}

请提取该论文的认知印象，按以下YAML格式输出（只输出YAML，不要其他文字）：

impression:
  one_sentence: "一句话总结这篇论文的核心贡献和你的评价"
  key_takeaway: "最值得记住的一个技术点"
  attitude: "excited|interested|neutral|skeptical_but_interested|skeptical|critical 之一"
  relevance_to_me: "high|medium|low 之一"
memorable_details:
  - "具体的数据点、图表发现或实验结果（引用具体数字）"
  - "（可有多条，最多5条）"
connections: []
figure_impressions: []
"""


def extract_pdf_text(pdf_path: str, max_chars: int = 8000) -> str:
    """Extract text from PDF, truncated to max_chars."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not HAS_PYPDF:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(parts)
    return text[:max_chars]


def generate_impression(client, model: str, paper_text: str,
                        metadata: dict) -> dict:
    """Use LLM to generate a cognitive impression from paper text."""
    prompt = IMPRESSION_PROMPT.format(
        paper_text=paper_text[:6000],
        title=metadata.get("title", "Unknown"),
        year=metadata.get("year", "Unknown"),
    )
    response = client.simple_chat(prompt, max_tokens=800)
    raw = response

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:yaml)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {"impression": {"one_sentence": raw[:200],
                               "key_takeaway": "", "attitude": "neutral",
                               "relevance_to_me": "medium"},
                "memorable_details": [], "connections": []}


def process_paper(client, model: str, pdf_path: str, memory_dir: str,
                  metadata: dict) -> str:
    """Full pipeline: PDF → impression YAML saved to memory/papers/."""
    # Extract text
    text = extract_pdf_text(pdf_path)

    # Generate impression
    impression_data = generate_impression(client, model, text, metadata)

    # Build full record
    record = {
        "source": {
            "title": metadata.get("title", "Unknown"),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "field": metadata.get("field", []),
            "pdf_path": pdf_path,
        },
        **impression_data,
        "ingested_at": datetime.now().strftime("%Y-%m-%d"),
    }

    # Generate filename from title and year
    slug = re.sub(r"[^\w\s-]", "", metadata.get("title", "paper").lower())
    slug = re.sub(r"\s+", "-", slug.strip())[:40]
    year = metadata.get("year", "0000")
    filename = f"{year}-{slug}.yaml"

    papers_dir = os.path.join(memory_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    output_path = os.path.join(papers_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(record, f, allow_unicode=True, default_flow_style=False)

    return output_path