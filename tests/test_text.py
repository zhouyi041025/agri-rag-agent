from agri_agent.text import char_ngrams, normalize, split_sentences, tokenize


def test_normalize_fullwidth_and_whitespace():
    assert normalize("芦荟　　pH ６.０") == "芦荟 pH 6.0"
    assert normalize("  a\r\nb  ") == "a\nb"


def test_tokenize_mixed_language():
    tokens = tokenize("YOLOv11 模型 mAP@50 达到 94.5%")
    assert "yolov11" in tokens
    assert "94.5" in tokens
    assert any("模型" in token for token in tokens)


def test_tokenize_drops_stopwords():
    assert "的" not in tokenize("芦荟的叶片")
    assert "的" in tokenize("芦荟的叶片", drop_stopwords=False)


def test_char_ngrams_for_chinese():
    grams = char_ngrams("炭疽病", 2)
    assert "炭疽" in grams and "疽病" in grams


def test_split_sentences_keeps_boundaries():
    sentences = split_sentences("第一句。第二句！第三句？")
    # normalize 会做 NFKC 归一化，全角 ！？ 会被转成半角
    assert sentences == ["第一句。", "第二句!", "第三句?"]
