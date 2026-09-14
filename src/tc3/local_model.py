"""Inferência portátil com o mesmo adaptador PEFT usado no treinamento."""
import json
import hashlib
from pathlib import Path

from tc3.prompts import messages_for


def choose_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS indisponível. Execute no Mac Apple Silicon ou escolha cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível.")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("Dispositivo inválido.")
    return requested


def load_adapter_manifest(path, model):
    path = Path(path)
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("model") != model:
        raise ValueError("Adaptador incompleto ou modelo base incompatível.")
    if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
        raise ValueError("Arquivos do adaptador ausentes.")
    if hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest() != manifest.get("adapter_sha256"):
        raise ValueError("Hash do adaptador divergente.")
    return manifest


class LocalGenerator:
    def __init__(self, model, adapter=None, device="auto", max_new_tokens=768, revision=None):
        # Imports pesados só ocorrem quando o backend local é selecionado.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = choose_device(torch, device)
        self.max_new_tokens = max_new_tokens
        self.model_id = model
        self.adapter = adapter
        if adapter:
            revision = load_adapter_manifest(adapter, model).get("model_revision")
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.float32,
            trust_remote_code=False, attn_implementation="eager",
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.to(self.device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", revision)

    def __call__(self, payload):
        prompt = self.tokenizer.apply_chat_template(
            messages_for(payload), tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_length = batch["input_ids"].shape[1]
        if input_length + self.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Contexto excede a capacidade do modelo; não será truncado.")
        with self.torch.inference_mode():
            output = self.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
        self.last_text = text
        # Falha explícita se houver texto fora do JSON. Sem reparação silenciosa.
        return json.loads(text)
