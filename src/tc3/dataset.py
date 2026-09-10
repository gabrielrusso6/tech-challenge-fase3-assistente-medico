"""Curadoria inicial e análise descritiva sem bibliotecas externas.

Não constitui anonimização clínica validada. Detectores são preventivos e limitados.
"""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def redact(text):
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF]", text)
    return text


def prepare(rows):
    cleaned, seen, groups = [], set(), {}
    for row in rows:
        if row.get("synthetic") is not True:
            raise ValueError("Esta versão aceita apenas dados declarados sintéticos.")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Split inválido.")
        if not row.get("group_id") or not row.get("source_ids"):
            raise ValueError("Grupo e fontes são obrigatórios.")
        if not row.get("question", "").strip() or not row.get("answer", "").strip():
            raise ValueError("Pergunta e resposta não podem estar vazias.")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Grupo compartilhado entre splits: possível vazamento.")
        groups[group] = row["split"]
        question, answer = redact(row["question"]), redact(row["answer"])
        key = hashlib.sha256(normalize(question).encode()).hexdigest()
        if key in seen:
            raise ValueError("Pergunta duplicada após normalização e remoção de identificadores.")
        seen.add(key)
        cleaned.append({**row, "question": question, "answer": answer,
                        "messages": [{"role": "system", "content":
                                      "Simulação acadêmica. Use fontes e exija revisão humana."},
                                     {"role": "user", "content": question},
                                     {"role": "assistant", "content": answer}]})
    return cleaned


def analyze(rows):
    if not rows:
        raise ValueError("Dataset vazio.")
    words = [len(r["question"].split()) + len(r["answer"].split()) for r in rows]
    return {"records": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
            "categories": dict(Counter(r["category"] for r in rows)),
            "unique_groups": len({r["group_id"] for r in rows}),
            "sources": dict(Counter(s for r in rows for s in r["source_ids"])),
            "words": {"min": min(words), "median": statistics.median(words), "max": max(words)},
            "limitations": ["Amostra inicial pequena; não demonstra qualidade clínica.",
                            "Contagem de palavras não equivale a tokens do modelo.",
                            "Sem revisão médica; protocolos sintéticos administrativos.",
                            "Sem análise semântica de duplicação nesta versão."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/seed_qa.json"))
    p.add_argument("--output", type=Path, default=Path("runtime/dataset"))
    args = p.parse_args()
    rows = prepare(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        records = [{"messages": r["messages"]} for r in rows if r["split"] == split]
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report = analyze(rows)
    (args.output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
