"""同义词表外置（data/synonyms.json）后的加载与扩展行为。"""

import pytest

from agri_agent.rag.expand import SYNONYMS, expand_query, load_synonyms


def test_synonyms_are_loaded_from_data_file():
    assert SYNONYMS["打药"] == ("施药", "喷药")
    assert len(SYNONYMS) >= 20


def test_expand_query_appends_synonyms():
    expanded = expand_query("打药")
    assert expanded.startswith("打药")
    assert "施药" in expanded


def test_load_synonyms_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_synonyms(tmp_path / "no-such-file.json")
