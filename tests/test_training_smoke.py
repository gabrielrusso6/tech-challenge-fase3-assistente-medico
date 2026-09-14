"""Treino real de um Qwen minúsculo aleatório, sem download e sem alegar qualidade médica."""
import copy

import pytest

torch = pytest.importorskip("torch", reason="Extras de treino testados no job training-smoke")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from tc3.train import optimize_step


def test_lora_updates_saves_and_reloads(tmp_path):
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        attention_dropout=0.0,
    )
    base = Qwen2ForCausalLM(config)
    clean_base = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"],
        r=4, lora_alpha=8, lora_dropout=0.0,
    ))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 2]])
    labels = ids.clone()
    labels[:, :3] = -100
    model.train()
    loss = optimize_step(model, optimizer, {"input_ids": ids, "labels": labels}, torch)
    assert loss > 0
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = PeftModel.from_pretrained(clean_base, tmp_path)
    loaded.eval()
    with torch.no_grad():
        actual = loaded(input_ids=ids).logits
    assert torch.allclose(expected, actual, atol=1e-5)
