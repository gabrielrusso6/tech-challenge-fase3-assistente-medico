"""Integração: grafo real com dublê somente na geração de linguagem."""
import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from tc3.generation import fixture_response, make_chain, validate_draft
from tc3.storage import Repository
from tc3.workflow import build_graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    repo = Repository(tmp_path / "demo.sqlite")
    repo.seed(ROOT / "data/patients.json")
    protocols = json.loads((ROOT / "data/protocols.json").read_text(encoding="utf-8"))
    return repo, protocols


def start(setup, chain=None, patient="SYN-001", question="Quais pendências?"):
    repo, protocols = setup
    graph = build_graph(repo, protocols, chain or make_chain("fixture"), "fixture")
    run = str(uuid4())
    config = {"configurable": {"thread_id": run}}
    state = graph.invoke({"run_id": run, "patient_id": patient, "question": question}, config)
    return graph, state, config


def test_pauses_without_writing_decision(setup):
    _, state, _ = start(setup)
    assert state["status"] == "awaiting_review"
    assert state["pending"] == ["Laudo de avaliação"]
    assert "__interrupt__" in state
    with setup[0].connect() as db:
        assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0


@pytest.mark.parametrize("approved,status", [(True, "approved"), (False, "rejected")])
def test_real_human_resume(setup, approved, status):
    graph, state, config = start(setup)
    result = graph.invoke(Command(resume={"approved": approved, "reviewer": "test-reviewer"}), config)
    assert result["status"] == status
    assert setup[0].events(state["run_id"])[-1]["event"] == "human_decision"
    with setup[0].connect() as db:
        assert db.execute("SELECT approved FROM decisions").fetchone()[0] == int(approved)


@pytest.mark.parametrize("decision", [True, {"approved": "yes", "reviewer": "x"},
                                     {"approved": True, "reviewer": ""}])
def test_invalid_approval_is_blocked(setup, decision):
    graph, _, config = start(setup)
    assert graph.invoke(Command(resume=decision), config)["status"] == "blocked"
    with setup[0].connect() as db:
        assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0


@pytest.mark.parametrize("patient", ["missing", "' OR 1=1 --"])
def test_missing_patient_never_reaches_generation(setup, patient):
    def unexpected(_):
        pytest.fail("Não deveria consultar gerador.")
    _, state, _ = start(setup, RunnableLambda(unexpected), patient=patient)
    assert state["status"] == "blocked"


@pytest.mark.parametrize("question", ["", " " * 4, "x" * 2001])
def test_invalid_question(setup, question):
    assert start(setup, question=question)[1]["status"] == "blocked"


def test_model_failure_does_not_become_fixture_success(setup):
    def unavailable(_):
        raise ConnectionError("offline")
    state = start(setup, RunnableLambda(unavailable))[1]
    assert state["status"] == "blocked"
    assert "draft" not in state


@pytest.mark.parametrize("change", ["fake_source", "changed_quote", "prescribe", "extra_text"])
def test_untrusted_generation_blocked(setup, change):
    def bad(payload):
        raw = fixture_response(payload)
        if change == "fake_source":
            raw["evidence"][0]["source_id"] = "INVENTED"
        elif change == "changed_quote":
            raw["evidence"][0]["quote"] = "Diagnóstico confirmado."
        elif change == "prescribe":
            raw["action"] = "prescribe"
        else:
            raw["prescription"] = "texto não permitido"
        return raw
    assert start(setup, RunnableLambda(bad))[1]["status"] == "blocked"


def test_no_cross_patient_context(setup):
    state = start(setup, patient="SYN-002")[1]
    ids = {s["id"] for s in state["sources"]}
    assert "PATIENT:SYN-002" in ids and "PATIENT:SYN-001" not in ids
    assert state["pending"] == []


def test_no_question_plaintext_in_audit(setup):
    marker = "MARCADOR-NAO-LOGAR"
    state = start(setup, question=marker)[1]
    assert marker not in json.dumps(setup[0].events(state["run_id"]))


def test_ollama_requires_explicit_model():
    with pytest.raises(ValueError):
        make_chain("ollama")


def test_missing_patient_evidence_rejected():
    with pytest.raises(ValueError):
        validate_draft({"action": "review_records", "evidence": [
            {"source_id": "SIM-SEG-001", "quote": "test"}]},
            [{"id": "SIM-SEG-001", "text": "test"}])
