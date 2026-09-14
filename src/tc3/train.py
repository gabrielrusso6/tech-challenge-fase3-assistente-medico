"""Fine-tuning LoRA real; execute --dry-run sem baixar modelo."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

from tc3.corpus import validate
from tc3.local_model import choose_device


def load_corpus(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("train", "validation", "test"):
        name = f"{split}_cases.jsonl"
        data = (path / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError(f"Hash divergente: {name}. Regenere o corpus.")
        part = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if any(r["split"] != split for r in part):
            raise ValueError("Split interno não corresponde ao arquivo.")
        rows.extend(part)
    validate(rows)
    return rows, manifest


def tokenize_example(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
                                            add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix)] != prefix:
        raise ValueError("Template/tokenização sem prefixo estável; máscara de resposta insegura.")
    if len(ids) > max_length:
        raise ValueError(f"Exemplo {row['id']} tem {len(ids)} tokens, limite {max_length}. Sem truncamento.")
    if len(ids) <= len(prefix):
        raise ValueError("Resposta sem tokens treináveis.")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):]}


def optimize_step(model, optimizer, batch, torch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise ValueError("Loss não finita; treinamento interrompido.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("runtime/corpus"))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", type=Path, default=Path("models/tc3-lora"))
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.rank < 1 or args.max_length < 2 or args.learning_rate <= 0:
        p.error("Hiperparâmetros devem ser positivos.")
    rows, manifest = load_corpus(args.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "validation"]
    plan = {"mode": "dry-run" if args.dry_run else "execute", "model": args.model,
            "steps": args.steps, "train_records": len(train_rows),
            "validation_records": len(valid_rows), "test_used_for_optimization": False,
            "corpus_sha256": manifest["corpus_sha256"]}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Diretório de saída não vazio. Use outro --output para preservar o experimento.")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=False)
    training = [tokenize_example(tokenizer, r, args.max_length) for r in train_rows]
    validation = [tokenize_example(tokenizer, r, args.max_length) for r in valid_rows]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
        trust_remote_code=False, attn_implementation="eager",
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"],
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    info = {
        **plan, "status": "running", "model_revision": resolved_revision,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": device, "platform": platform.platform(), "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n)
                     for n in ("torch", "transformers", "peft", "accelerate")},
        "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
        "total_parameters": sum(v.numel() for v in model.parameters()),
        "max_train_tokens": max(len(x["input_ids"]) for x in training),
    }
    write_json(args.output / "run.json", info)
    def batch(example):
        return {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in example.items()}
    def validation_loss():
        model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for ex in validation:
                n = sum(t != -100 for t in ex["labels"][1:])
                loss = float(model(**batch(ex)).loss.detach().cpu())
                if not math.isfinite(loss):
                    raise ValueError("Validation loss não finita.")
                total += loss * n
                tokens += n
        return total / tokens
    started = time.perf_counter()
    try:
        before = validation_loss()
        order = list(range(len(training)))
        with (args.output / "training.jsonl").open("w", encoding="utf-8") as log:
            for step in range(args.steps):
                if step % len(order) == 0:
                    rng.shuffle(order)
                model.train()
                loss = optimize_step(model, optimizer, batch(training[order[step % len(order)]]), torch)
                event = {"step": step + 1, "loss": loss, "elapsed_seconds": time.perf_counter() - started}
                log.write(json.dumps(event) + "\n")
                log.flush()
                if step == 0 or (step + 1) % 10 == 0:
                    print(json.dumps(event), flush=True)
        after = validation_loss()
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        info.update(status="completed", validation_loss_before=before, validation_loss_after=after,
                    elapsed_seconds=time.perf_counter() - started,
                    completed_steps=args.steps)
        if device == "cuda":
            info["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        elif device == "mps":
            info["end_allocated_bytes_not_peak"] = torch.mps.current_allocated_memory()
        info["adapter_sha256"] = hashlib.sha256(
            (args.output / "adapter_model.safetensors").read_bytes()).hexdigest()
    except Exception as exc:
        info.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output / "run.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
