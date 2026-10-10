"""The output holds its shape: schema validation + repair + degrade."""

from __future__ import annotations

from app.schemas import LLMClassification
from app.triage import _parse_classification


def test_valid_json_parses_clean():
    raw = (
        '{"category":"billing_question","severity":"low",'
        '"reply_draft":"hi","detected_language":"english",'
        '"refund_hint":"none","confidence":0.8,"reasoning":"ok"}'
    )
    obj, repaired = _parse_classification(raw)
    assert isinstance(obj, LLMClassification)
    assert repaired is False


def test_invented_category_is_rejected():
    raw = (
        '{"category":"please_give_money","severity":"low",'
        '"reply_draft":"hi","confidence":0.8}'
    )
    obj, _ = _parse_classification(raw)
    assert obj is None  # enum guard refuses the invented category


def test_fenced_json_is_repaired():
    raw = (
        "```json\n"
        '{"category":"cancellation","severity":"low","reply_draft":"ok",'
        '"detected_language":"english","refund_hint":"none","confidence":0.7,'
        '"reasoning":"r"}\n'
        "```"
    )
    obj, repaired = _parse_classification(raw)
    assert isinstance(obj, LLMClassification)
    assert repaired is True


def test_prose_is_unrecoverable():
    raw = "Sure! Here is my analysis of the ticket. The user seems upset..."
    obj, _ = _parse_classification(raw)
    assert obj is None  # caller degrades to rules


def test_confidence_out_of_range_rejected():
    raw = (
        '{"category":"other","severity":"low","reply_draft":"x",'
        '"confidence":9.0}'
    )
    obj, _ = _parse_classification(raw)
    assert obj is None
