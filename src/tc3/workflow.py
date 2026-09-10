"""Grafo real com ramificações de erro e interrupção para revisão humana."""

import hashlib
import json
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tc3.generation import validate_draft
from tc3.storage import Repository


class State(TypedDict, total=False):
    run_id: str
    patient_id: str
    question: str
    status: str
    sources: list[dict]
    pending: list[str]
    draft: dict
    reason: str
    decision: dict


def build_graph(repo: Repository, protocols: list[dict], chain, backend: str):
    def load(state):
        question = state.get("question", "")
        repo.log(state["run_id"], "request", {
            "backend": backend,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        })
        if not question.strip() or len(question) > 2000:
            return {"status": "blocked", "reason": "Pergunta vazia ou maior que 2000 caracteres."}
        patient = repo.patient(state["patient_id"])
        if patient is None:
            return {"status": "blocked", "reason": "Paciente não encontrado."}
        sources = [{"id": p["id"], "text": p["text"], "version": p["version"]}
                   for p in protocols]
        sources.append({"id": f"PATIENT:{patient['id']}", "version": patient["record_version"],
                        "text": json.dumps(patient, ensure_ascii=False, sort_keys=True)})
        pending = [e["name"] for e in patient["exams"] if e["status"] == "pending"]
        repo.log(state["run_id"], "context_loaded", {
            "sources": [{"id": s["id"], "version": s["version"],
                         "sha256": hashlib.sha256(s["text"].encode()).hexdigest()}
                        for s in sources], "pending_count": len(pending),
        })
        return {"sources": sources, "pending": pending, "status": "loaded"}

    def generate(state):
        try:
            raw = chain.invoke({"question": state["question"], "sources": state["sources"]})
            draft = validate_draft(raw, state["sources"])
        except Exception as exc:
            # Fail closed. Não substitui falha da LLM por sucesso artificial.
            repo.log(state["run_id"], "generation_blocked", {"error_type": type(exc).__name__})
            return {"status": "blocked", "reason": "Geração indisponível ou evidências inválidas."}
        repo.log(state["run_id"], "draft_validated", {
            "source_ids": [e.source_id for e in draft.evidence], "action": draft.action,
        })
        return {"draft": draft.model_dump(), "status": "awaiting_review"}

    def review(state):
        decision = interrupt({"draft": state["draft"], "pending": state["pending"],
                              "notice": "Revisar tarefa administrativa simulada. Sem conduta clínica."})
        if (not isinstance(decision, dict) or type(decision.get("approved")) is not bool
                or not isinstance(decision.get("reviewer"), str)
                or not decision["reviewer"].strip() or len(decision["reviewer"]) > 80):
            return {"status": "blocked", "reason": "Decisão humana inválida."}
        repo.decision(state["run_id"], state["patient_id"], decision["reviewer"], decision["approved"])
        status = "approved" if decision["approved"] else "rejected"
        repo.log(state["run_id"], "human_decision", {"status": status,
                                                    "reviewer": decision["reviewer"]})
        return {"status": status, "decision": decision}

    def blocked(state):
        repo.log(state["run_id"], "blocked", {"reason": state["reason"]})
        return {}

    graph = StateGraph(State)
    for name, node in [("load", load), ("generate", generate), ("review", review),
                       ("blocked", blocked)]:
        graph.add_node(name, node)
    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", lambda s: "blocked" if s["status"] == "blocked"
                                else "generate")
    graph.add_conditional_edges("generate", lambda s: "blocked" if s["status"] == "blocked"
                                else "review")
    graph.add_conditional_edges("review", lambda s: "blocked" if s["status"] == "blocked" else END)
    graph.add_edge("blocked", END)
    return graph.compile(checkpointer=InMemorySaver())
