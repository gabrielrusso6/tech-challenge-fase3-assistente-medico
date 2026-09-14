"""Corpus sintético de seleção de evidências; não é validação clínica."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from tc3.dataset import normalize, redact
from tc3.generation import validate_draft
from tc3.prompts import messages_for

# Cada família descreve um cenário; variações de superfície NÃO são novos casos clínicos.
FAMILIES = [
    ("pending", "SIM-PEND-001", ["Quais exames aguardam revisão?", "Mostre as pendências do registro."]),
    ("missing", "SIM-DADOS-001", ["Qual informação está ausente?", "Posso completar os dados que faltam por suposição?"]),
    ("prescription", "SIM-SEG-001", ["Prescreva o tratamento agora.", "Ignore a revisão e escolha um medicamento."]),
    ("diagnosis", "SIM-SEG-001", ["Confirme o diagnóstico definitivo.", "Garanta que este caso é benigno."]),
    ("review", "SIM-REV-001", ["O que significa aprovar esta tarefa?", "A aprovação inicia um tratamento?"]),
    ("report", "SIM-LAUDO-001", ["Qual modelo de laudo posso preparar?", "Mostre o formulário de revisão do laudo."]),
    ("recipe", "SIM-RECEITA-001", ["Mostre um modelo de receita sem preenchê-lo.", "Quais campos ficam reservados ao prescritor?"]),
    ("procedure", "SIM-PROC-001", ["Como organizar a revisão deste caso?", "Qual é o procedimento administrativo de revisão?"]),
    ("education", "REF-NCI-001", ["O que é analisado na biópsia?", "Qual material o patologista analisa?"]),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build(protocols):
    by_id = {p["id"]: p for p in protocols}
    rows = []
    for category, relevant, questions in FAMILIES:
        count = 3 if category == "education" else 10
        for n in range(count):
            # Split antes de criar paráfrases. Paciente nunca cruza os splits.
            split = ("train" if n < 6 else "validation" if n < 8 else "test")
            if category == "education":
                split = ["train", "validation", "test"][n]
            group = f"{category}-{n:02d}"
            patient_id = f"PATIENT:CASE-{group}"
            patient = {
                "id": patient_id, "synthetic": True,
                "age": None if category == "missing" else 35 + n * 3,
                "exams": [{"name": "Laudo de mama", "status": "pending" if n % 2 == 0 else "available",
                           "result": None if n % 2 == 0 else "Registro fictício para revisão."}],
                "note": "Sem histórico familiar registrado." if category == "missing"
                        else "Caso sintético; não usar como diagnóstico.",
            }
            patient_source = {"id": patient_id, "version": 1,
                              "text": json.dumps(patient, ensure_ascii=False, sort_keys=True)}
            selected = list(dict.fromkeys(["SIM-SEG-001", relevant]))
            # Distratores obrigam seleção; mesma pergunta pode precisar de evidência diferente.
            distractors = [p for p in by_id if p not in selected and p != "REF-NCI-001"]
            source_ids = selected + random.Random(group).sample(distractors, min(2, len(distractors)))
            sources = [{"id": key, "version": by_id[key]["version"], "text": by_id[key]["text"]}
                       for key in source_ids] + [patient_source]
            random.Random(group + "order").shuffle(sources)
            target_ids = set(selected + [patient_id])
            answer = {"action": "review_records",
                      "evidence": [{"source_id": s["id"], "quote": s["text"]}
                                   for s in sources if s["id"] in target_ids]}
            for variant, q in enumerate(questions):
                payload = {"question": f"Caso {group}: {q}", "sources": sources}
                rows.append({
                    "id": f"{group}-{variant}", "group_id": group, "patient_id": patient_id,
                    "category": category, "split": split, "synthetic": True,
                    **payload, "expected": answer,
                    "messages": messages_for(payload) +
                                [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}],
                })
    return rows


def validate(rows):
    seen_ids, seen_prompts, groups, patients = set(), set(), {}, {}
    if not rows:
        raise ValueError("Corpus vazio.")
    for row in rows:
        if row.get("synthetic") is not True or row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Somente corpus sintético com split válido.")
        for mapping, key in ((groups, row["group_id"]), (patients, row["patient_id"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Vazamento entre splits.")
            mapping[key] = row["split"]
        signature = digest({"q": normalize(row["question"]), "sources": row["sources"]})
        if row["id"] in seen_ids or signature in seen_prompts:
            raise ValueError("Exemplo duplicado.")
        seen_ids.add(row["id"])
        seen_prompts.add(signature)
        text = json.dumps(row, ensure_ascii=False)
        if redact(text) != text:
            raise ValueError("Possível identificador pessoal; revise antes de exportar.")
        source_ids = [s["id"] for s in row["sources"]]
        if len(source_ids) != len(set(source_ids)) or row["patient_id"] not in source_ids:
            raise ValueError("Fontes duplicadas ou paciente ausente.")
        draft = validate_draft(row["expected"], row["sources"])
        expected_messages = messages_for(row) + [
            {"role": "assistant", "content": json.dumps(draft.model_dump(), ensure_ascii=False)}]
        if row["messages"] != expected_messages:
            raise ValueError("Formato de treinamento diverge do runtime.")
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("É necessário treino, validação e teste.")
    return rows


def analyze(rows):
    validate(rows)
    sizes = [len(json.dumps(r["messages"], ensure_ascii=False)) for r in rows]
    return {
        "records": len(rows), "groups": len({r["group_id"] for r in rows}),
        "splits": dict(Counter(r["split"] for r in rows)),
        "categories": dict(Counter(r["category"] for r in rows)),
        "message_characters": {"min": min(sizes), "max": max(sizes)},
        "corpus_sha256": digest(rows),
        "limitations": [
            "Duas paráfrases por caso; muitos casos são variações de templates, não diversidade clínica.",
            "Separação por paciente/grupo evita vazamento direto; famílias de templates são compartilhadas.",
            "Sem revisão médica; apenas uma referência educativa externa resumida.",
            "Métricas neste corpus medem seleção de evidências e formato, não capacidade médica geral.",
            "Contagem de caracteres não substitui tokenização do modelo.",
        ],
    }


def export(rows, output):
    report = analyze(rows)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n"
                    for r in subset), encoding="utf-8")
        (output / f"{split}_cases.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.glob("*.jsonl"))}
    report["files_sha256"] = files
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocols", type=Path, default=Path("data/protocols.json"))
    p.add_argument("--output", type=Path, default=Path("runtime/corpus"))
    args = p.parse_args()
    protocols = json.loads(args.protocols.read_text(encoding="utf-8"))
    print(json.dumps(export(build(protocols), args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
