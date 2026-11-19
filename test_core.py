from core import score_index, SCORE_KEYS, VALIDITY_KEYS, parse_json_response, clamp_score_map, clean_source_ids


def test_score_index():
    result = score_index({k: 1.0 for k in SCORE_KEYS}, SCORE_KEYS)
    assert result["score"] == 5.0
    assert result["evaluated"] == 5


def test_score_index_holds_with_too_few():
    result = score_index({"logic": 1.0, "headline_match": 0.5}, VALIDITY_KEYS)
    assert result["score"] is None
    assert result["status"] == "평가 보류"


def test_parse_json_with_fence():
    assert parse_json_response('```json\n{"ok": true}\n```')["ok"] is True


def test_clamp_score_map():
    cleaned = clamp_score_map({"logic": 3, "context": 0.5}, VALIDITY_KEYS)
    assert cleaned["logic"] is None
    assert cleaned["context"] == 0.5


def test_source_id_filter():
    assert clean_source_ids(["S1", "S9", "http://fake"], {"S1", "S2"}) == ["S1"]
