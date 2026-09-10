import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from tc3.generation import make_chain
from tc3.storage import Repository
from tc3.workflow import build_graph


def main():
    parser = argparse.ArgumentParser(description="TC3 — simulação acadêmica; sem uso clínico")
    parser.add_argument("--mode", choices=["fixture", "ollama"], required=True)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=Path("runtime/demo.sqlite"))
    parser.add_argument("--patient", default="SYN-001")
    parser.add_argument("--question", default="Quais registros estão pendentes para revisão?")
    parser.add_argument("--reviewer", default="avaliador-demo")
    args = parser.parse_args()
    try:
        chain = make_chain(args.mode, args.model, os.getenv("OLLAMA_BASE_URL"))
        repo = Repository(args.db)
        repo.seed(args.data_dir / "patients.json")
        protocols = json.loads((args.data_dir / "protocols.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    backend = args.mode if args.mode == "fixture" else f"ollama:{args.model}"
    print(f"Modo: {backend}. Fixture NÃO usa LLM; Ollama NÃO implica fine-tuning.")
    graph = build_graph(repo, protocols, chain, backend)
    run_id = str(uuid4())
    config = {"configurable": {"thread_id": run_id}}
    result = graph.invoke({"run_id": run_id, "patient_id": args.patient,
                           "question": args.question}, config)
    if "__interrupt__" in result:
        print(json.dumps(result["__interrupt__"][0].value, ensure_ascii=False, indent=2))
        try:
            answer = input("Aprovar tarefa simulada? [s/n; Enter deixa pendente]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in {"s", "n"}:
            result = graph.invoke(Command(resume={"approved": answer == "s",
                                                  "reviewer": args.reviewer}), config)
        else:
            print("Sem decisão. Checkpoint apenas em memória; reinicie o caso ao executar novamente.")
    print(json.dumps({"run_id": run_id, "status": result["status"],
                      "reason": result.get("reason"), "audit": repo.events(run_id)},
                     ensure_ascii=False, indent=2))
    if result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
