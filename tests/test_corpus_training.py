import copy
import json
from pathlib import Path

import pytest

from tc3.corpus import analyze, build, export, validate
from tc3.evaluate_llm import score
from tc3.local_model import load_adapter_manifest
from tc3.prompts import messages_for
from tc3.train import load_corpus, tokenize_example

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows():
    return build(json.loads((ROOT / "data/protocols.json").read_text(encoding="utf-8")))


def test_all_targets_match_runtime(rows):
    assert len(validate(rows)) == 166
    assert analyze(rows)["splits"] == {"train": 98, "validation": 34, "test": 34}
    for row in rows:
        assert row["messages"][:-1] == messages_for(row)


def test_paraphrases_never_cross_splits(rows):
    seen = {}
    for row in rows:
        if row["group_id"] in seen:
            assert seen[row["group_id"]] == row["split"]
        seen[row["group_id"]] = row["split"]


@pytest.mark.parametrize("mutation", ["group", "patient", "answer", "duplicate", "pii", "messages"])
def test_invalid_corpus_rejected(rows, mutation):
    rows = copy.deepcopy(rows)
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    if mutation == "group":
        test["group_id"] = train["group_id"]
    elif mutation == "patient":
        test["patient_id"] = train["patient_id"]
    elif mutation == "answer":
        test["expected"]["evidence"][0]["quote"] = "Conteúdo inventado"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "pii":
        test["question"] += " fake@example.com"
    else:
        test["messages"][0]["content"] = "Contrato divergente"
    with pytest.raises(ValueError):
        validate(rows)


def test_export_integrity_and_tamper_rejection(rows, tmp_path):
    manifest = export(rows, tmp_path)
    loaded, _ = load_corpus(tmp_path)
    assert len(loaded) == manifest["records"]
    with (tmp_path / "train_cases.jsonl").open("a", encoding="utf-8") as fp:
        fp.write("\n")
    with pytest.raises(ValueError, match="Hash"):
        load_corpus(tmp_path)


def test_evaluation_penalizes_missing_relevant_source(rows):
    row = next(r for r in rows if r["category"] == "pending")
    raw = copy.deepcopy(row["expected"])
    raw["evidence"] = [e for e in raw["evidence"] if e["source_id"] != "SIM-PEND-001"]
    metrics = score(json.dumps(raw), row)
    assert metrics["valid"] and metrics["precision"] == 1
    assert metrics["recall"] < 1 and not metrics["exact_sources"]


def test_evaluation_rejects_hallucination(rows):
    raw = copy.deepcopy(rows[0]["expected"])
    raw["evidence"][0]["quote"] = "Tratamento inventado"
    assert not score(json.dumps(raw), rows[0])["valid"]


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        value = "".join(m["role"] + ":" + m["content"] + "|" for m in messages)
        return value + ("assistant:" if add_generation_prompt else "")
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def test_only_answer_tokens_contribute_to_loss(rows):
    row = rows[0]
    tokenized = tokenize_example(CharTokenizer(), row, 10000)
    prefix_len = len(CharTokenizer().apply_chat_template(
        row["messages"][:-1], add_generation_prompt=True))
    assert tokenized["labels"][:prefix_len] == [-100] * prefix_len
    assert tokenized["labels"][prefix_len:] == tokenized["input_ids"][prefix_len:]


def test_long_example_never_silently_truncated(rows):
    with pytest.raises(ValueError, match="Sem truncamento"):
        tokenize_example(CharTokenizer(), rows[0], 20)


def test_incomplete_adapter_rejected(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "running", "model": "x"}))
    with pytest.raises(ValueError):
        load_adapter_manifest(tmp_path, "x")


def test_comparison_rejects_different_corpus():
    from tc3.compare import compare
    base = {"model": "x", "corpus_sha256": "a"}
    tuned = {"model": "x", "corpus_sha256": "b"}
    with pytest.raises(ValueError, match="corpus_sha256"):
        compare(base, tuned)


def test_comparison_reports_negative_results():
    from tc3.compare import compare
    base = {"model": "x", "revision": "abc", "split": "test", "corpus_sha256": "a",
            "count": 2, "max_new_tokens": 768, "device": "cpu", "adapter": None,
            "valid_rate": 1.0, "exact_source_rate": 1.0, "mean_precision": 1.0,
            "mean_recall": 1.0, "mean_seconds": 2.0}
    tuned = {**base, "adapter": "models/adapter", "valid_rate": 0.5}
    assert compare(base, tuned)["valid_rate"]["delta"] == -0.5
