# tests/ingestion/test_paper_processor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from ingestion.paper_processor import extract_pdf_text, generate_impression, process_paper


def _make_fake_pdf(tmp_path) -> str:
    """Create a minimal valid PDF for testing."""
    # Use pypdf to create a test PDF with text
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        pdf_path = str(tmp_path / "test_paper.pdf")
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path
    except Exception:
        # fallback: write a minimal PDF bytes
        pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        pdf_path = str(tmp_path / "test_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        return pdf_path


def test_extract_pdf_text_returns_string(tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    text = extract_pdf_text(pdf_path)
    assert isinstance(text, str)


def test_extract_pdf_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_pdf_text("/nonexistent/paper.pdf")


def test_generate_impression_structure():
    """generate_impression should return a dict matching the schema."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
impression:
  one_sentence: "用MOF衍生碳负载单原子Pt，思路好但稳定性不足"
  key_takeaway: "单原子分散可降低Pt用量"
  attitude: skeptical_but_interested
  relevance_to_me: high
memorable_details:
  - "Fig.3的EXAFS数据漂亮"
connections: []
""")]
    mock_client.messages.create.return_value = mock_response

    result = generate_impression(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        paper_text="Abstract: We report single-atom Pt on MOF-derived carbon for HER...",
        metadata={"title": "SAC for HER", "year": 2024},
    )
    assert "impression" in result
    assert "one_sentence" in result["impression"]
    assert "memorable_details" in result


def test_process_paper_creates_yaml(tmp_path):
    """process_paper should write a YAML file to memory/papers/."""
    pdf_path = _make_fake_pdf(tmp_path)
    memory_dir = str(tmp_path / "memory")
    os.makedirs(os.path.join(memory_dir, "papers"), exist_ok=True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
impression:
  one_sentence: "测试论文印象"
  key_takeaway: "关键结论"
  attitude: neutral
  relevance_to_me: medium
memorable_details: []
connections: []
""")]
    mock_client.messages.create.return_value = mock_response

    output_path = process_paper(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        pdf_path=pdf_path,
        memory_dir=memory_dir,
        metadata={"title": "Test Paper", "authors": ["Test Author"], "year": 2024,
                  "field": ["hydrogen", "catalyst"]},
    )

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = yaml.safe_load(f)
    assert "source" in data
    assert "impression" in data
    assert data["source"]["title"] == "Test Paper"