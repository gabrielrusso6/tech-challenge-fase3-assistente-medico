"""Avaliação de gerações reais. Sem métricas artificiais ou fallback."""
import argparse
import json
import time
from pathlib import Path

from tc3.generation import validate_draft
from tc3.local_model import LocalGenerator
from tc3.train import load_corpus


def score(raw, case):
    try:
        parsed = json.loads(raw)
        valid = validate_draft(parsed, case["sources"])
    except (ValueError, TypeError):
        return {"valid": False, "precision": 0.0, "recall": 0.0, "exact_sources": False}
    found = {e.source_id for e in valid.evidence}
    expected = {e["source_id"] for e in case["expected"]["evidence"]}
    intersection = len(found & expected)
    return {"valid": True, "precision": intersection / len(found),
            "recall": intersection / len(expected), "exact_sources": found == expected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--revision", help="Fixar a mesma revisão do modelo nos dois experimentos")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--data", type=Path, default=Path("runtime/corpus"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    args = p.parse_args()
    rows, manifest = load_corpus(args.data)
    cases = [r for r in rows if r["split"] == args.split]
    if args.output.exists():
        p.error("Saída já existe; use outro nome para preservar o resultado.")
    generator = LocalGenerator(args.model, args.adapter, args.device, revision=args.revision)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for case in cases:
            generator.last_text = ""
            start = time.perf_counter()
            error = None
            try:
                generator(case)
            except Exception as exc:
                error = type(exc).__name__
            raw = generator.last_text
            metrics = score(raw, case) if error is None else {
                "valid": False, "precision": 0.0, "recall": 0.0, "exact_sources": False}
            row = {"case_id": case["id"], "category": case["category"], "raw": raw,
                   "error": error, "seconds": time.perf_counter() - start, **metrics}
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    summary = {
        "model": args.model, "revision": generator.revision, "adapter": args.adapter,
        "device": generator.device, "split": args.split, "count": len(results),
        "corpus_sha256": manifest["corpus_sha256"], "max_new_tokens": generator.max_new_tokens,
        "valid_rate": sum(r["valid"] for r in results) / len(results),
        "exact_source_rate": sum(r["exact_sources"] for r in results) / len(results),
        "mean_precision": sum(r["precision"] for r in results) / len(results),
        "mean_recall": sum(r["recall"] for r in results) / len(results),
        "mean_seconds": sum(r["seconds"] for r in results) / len(results),
        "limitation": "Mede schema e seleção extrativa; não é validação clínica.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
