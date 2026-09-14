"""Compara resultados realmente gerados sob condições equivalentes."""
import argparse
import json
from pathlib import Path


def compare(base, tuned):
    for key in ("model", "revision", "split", "corpus_sha256", "count", "max_new_tokens", "device"):
        if base.get(key) != tuned.get(key):
            raise ValueError(f"Comparação incompatível: {key}")
    if not base.get("revision"):
        raise ValueError("Revisão do modelo não registrada.")
    if base.get("adapter") or not tuned.get("adapter"):
        raise ValueError("Informe base sem adaptador e ajustado com adaptador.")
    return {k: {"base": base[k], "adjusted": tuned[k], "delta": tuned[k] - base[k]}
            for k in ("valid_rate", "exact_source_rate", "mean_precision",
                      "mean_recall", "mean_seconds")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("adjusted", type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.base.read_text(encoding="utf-8")),
                     json.loads(args.adjusted.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
