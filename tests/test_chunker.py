from agri_agent.rag.chunker import chunk_document, chunk_documents
from agri_agent.rag.loader import Document

SAMPLE = """# 测试文档

## 炭疽病

症状识别：叶片出现凹陷病斑，边缘黄褐色。

防治要点：发病初期喷施咪鲜胺，间隔七到十天一次。

## 褐斑病

症状识别：病斑近圆形，后期穿孔。
"""


def _document() -> Document:
    return Document(doc_id="demo", title="测试文档", text=SAMPLE.replace("\n", "\n").strip(), source="demo.md")


def test_heading_strategy_keeps_section_names():
    chunks = chunk_document(_document(), max_chars=400, overlap=50, strategy="heading")
    sections = {chunk.section for chunk in chunks}
    assert {"炭疽病", "褐斑病"} <= sections
    assert any("咪鲜胺" in chunk.text for chunk in chunks)


def test_same_section_paragraphs_are_merged():
    chunks = chunk_document(_document(), max_chars=400, overlap=50, strategy="heading")
    anthracnose = [chunk for chunk in chunks if chunk.section == "炭疽病"]
    assert len(anthracnose) == 1, "同一小节内的段落应合并，而不是按空行拆碎"
    assert "凹陷" in anthracnose[0].text and "咪鲜胺" in anthracnose[0].text


def test_fixed_strategy_has_no_section():
    chunks = chunk_document(_document(), max_chars=60, overlap=20, strategy="fixed")
    assert chunks and all(chunk.section == "" for chunk in chunks)
    assert all(len(chunk.text) <= 80 for chunk in chunks)


def test_chunk_ids_are_unique_and_display_text_has_header():
    chunks = chunk_documents([_document()], max_chars=120, overlap=30, strategy="heading")
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.title in chunk.display_text for chunk in chunks)
