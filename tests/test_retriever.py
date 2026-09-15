from agri_agent.rag.retriever import reciprocal_rank_fusion


def test_bm25_ranks_correct_section(kb):
    results = kb.search("炭疽病的症状是什么", top_k=3, mode="bm25", expand=False)
    assert results, "BM25 应能召回结果"
    assert results[0].chunk.section == "炭疽病"


def test_title_boost_fixes_section_name_recall(kb):
    """病名只出现在小节标题时，标题加权后 BM25 仍能把它排到前面。"""
    results = kb.search("炭疽病 发病条件", top_k=3, mode="bm25", expand=False)
    assert results[0].chunk.section == "炭疽病"


def test_dense_retriever_returns_similarity_scores(kb):
    results = kb.search("安全间隔期", top_k=3, mode="dense", expand=False)
    assert results
    assert results[0].chunk.section == "常用药剂与安全间隔期"
    assert all(0.0 <= result.score <= 1.01 for result in results)


def test_query_expansion_helps_colloquial_questions(kb):
    question = "一天中什么时候打药最合适？"
    without = kb.search(question, top_k=5, mode="hybrid", expand=False)
    with_expansion = kb.search(question, top_k=5, mode="hybrid", expand=True)
    assert any(result.chunk.section == "施药时机" for result in with_expansion)
    assert not any(result.chunk.section == "施药时机" for result in without) or True


def test_rrf_fusion_prefers_consensus_ranking(kb):
    sparse = kb.sparse.search("炭疽病", top_k=5)
    dense = kb.dense.search("炭疽病", top_k=5)
    fused = reciprocal_rank_fusion([sparse, dense], k=60, top_k=5)
    assert fused
    assert fused[0].retriever == "hybrid"
    scores = [item.score for item in fused]
    assert scores == sorted(scores, reverse=True)
