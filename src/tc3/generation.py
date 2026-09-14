"""Pipeline LangChain: saída extrativa para uma primeira base verificável."""

import json
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=1)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["review_records"]
    evidence: list[Evidence] = Field(min_length=1, max_length=12)


def fixture_response(payload: dict) -> dict:
    """Dublê explícito para testes de infraestrutura; não é modelo nem fine-tuning."""
    return {
        "action": "review_records",
        "evidence": [{"source_id": s["id"], "quote": s["text"]} for s in payload["sources"]],
    }


def make_chain(mode: str, model: str | None = None, base_url: str | None = None,
               adapter: str | None = None, device: str = "auto"):
    if mode == "fixture":
        return RunnableLambda(fixture_response)
    if mode == "local":
        if not model:
            raise ValueError("Modo local exige modelo base explícito.")
        from tc3.local_model import LocalGenerator
        return RunnableLambda(LocalGenerator(model, adapter=adapter, device=device))
    if mode != "ollama" or not model:
        raise ValueError("Modo ollama exige --model ou OLLAMA_MODEL explícito.")
    from langchain_ollama import ChatOllama

    from tc3.prompts import messages_for
    prepare = RunnableLambda(messages_for)
    kwargs = {"model": model, "temperature": 0, "format": "json", "num_predict": 1024,
              "client_kwargs": {"timeout": 60.0}}
    if base_url:
        kwargs["base_url"] = base_url
    return prepare | ChatOllama(**kwargs) | JsonOutputParser()


def validate_draft(raw: dict, sources: list[dict]) -> Draft:
    draft = Draft.model_validate(raw)
    allowed = {s["id"]: s["text"] for s in sources}
    used = set()
    for item in draft.evidence:
        # Igualdade completa evita citações truncadas que invertam o sentido de uma regra.
        if item.source_id not in allowed or item.quote != allowed[item.source_id]:
            raise ValueError("Fonte desconhecida ou citação alterada.")
        if item.source_id in used:
            raise ValueError("Fonte duplicada.")
        used.add(item.source_id)
    if "SIM-SEG-001" not in used or not any(s.startswith("PATIENT:") for s in used):
        raise ValueError("Faltam evidências obrigatórias de segurança ou do paciente.")
    return draft
