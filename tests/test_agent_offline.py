def test_knowledge_question_returns_answer_with_citations(agent):
    answer = agent.answer("芦荟炭疽病怎么防治？")
    assert "知识库回答" in answer.answer
    assert answer.citations, "回答必须带引用"
    assert all(item["doc"] for item in answer.citations)
    assert any(step["name"] == "search_knowledge" for step in answer.steps)


def test_citation_numbers_match_answer_markers(agent):
    answer = agent.answer("炭疽病怎么防治？")
    numbers = {item["index"] for item in answer.citations}
    assert numbers, "引用台账不应为空"
    for number in numbers:
        assert f"[{number}]" in answer.answer


def test_unknown_topic_is_refused_not_hallucinated(agent):
    answer = agent.answer("火星上怎么种芦荟")
    assert "暂未收录" in answer.answer
    assert "咪鲜胺" not in answer.answer


def test_dosage_question_routes_to_tool(agent):
    answer = agent.answer("我有 5 亩芦荟要防治炭疽病，用代森锰锌要多少量？")
    assert [step["name"] for step in answer.steps][0] == "calc_spray_dosage"
    assert "800.0 g" in answer.answer


def test_env_question_routes_to_env_tool(agent):
    answer = agent.answer("S3 现在田间环境怎么样？")
    assert [step["name"] for step in answer.steps][0] == "get_field_env"
    assert "监测点：S3" in answer.answer


def test_missing_area_asks_for_clarification(agent):
    answer = agent.answer("代森锰锌要用多少？")
    assert "施药面积" in answer.answer or "多少量" in answer.answer or "亩" in answer.answer


def test_on_step_callback_streams_tool_calls(agent):
    seen = []
    agent.answer("炭疽病怎么防治？", on_step=seen.append)
    assert seen and seen[0]["name"] == "search_knowledge"


def _history(question: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def test_follow_up_borrows_previous_question(agent):
    """省略式追问自身不含实词，应结合上一轮问题补全检索意图。"""
    first = agent.answer("芦荟炭疽病怎么防治？")
    answer = agent.answer("那这种药要打几次？", _history("芦荟炭疽病怎么防治？", first.answer))
    assert "多轮理解" in answer.answer
    assert "暂未收录" not in answer.answer
    assert "3 次" in answer.answer
    # 检索用的是补全后的意图，而不是追问本身
    assert answer.steps[0]["arguments"]["query"] == "芦荟炭疽病怎么防治？"
    assert answer.citations


def test_follow_up_with_own_topic_does_not_borrow(agent):
    """追问自带领域词说明换了话题，应独立检索而不是沿用上一轮。"""
    first = agent.answer("芦荟炭疽病怎么防治？")
    answer = agent.answer("那褐斑病呢？", _history("芦荟炭疽病怎么防治？", first.answer))
    assert "多轮理解" not in answer.answer
    assert "暂未收录" not in answer.answer
    assert answer.steps[0]["arguments"]["query"] == "那褐斑病呢？"
    # 命中的是「褐斑病」小节（上一轮命中的是农药文档），说明确实换了话题
    assert any(item["section"] == "褐斑病" for item in answer.citations)


def test_follow_up_without_history_degrades_gracefully(agent):
    """没有上下文时宁可弃答，也不能猜。"""
    answer = agent.answer("那这种药要打几次？")
    assert "暂未收录" in answer.answer
