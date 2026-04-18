# Unsloth Cheatsheet

Quick reference for fine-tuning medical LLMs with Unsloth.

---

## Installation

```bash
# Recommended (fastest, handles CUDA deps)
curl -fsSL https://unsloth.ai/install.sh | sh

# Or via uv (preferred on RunPod)
uv pip install unsloth

# Or via pip (slower dep resolution)
pip install unsloth

# Verify
python -c "import unsloth; print(unsloth.__version__)"

# Check CUDA version before installing (critical for version pinning)
python -c "import torch; print(torch.version.cuda)"
```

**Qwen3.5 requirement**: Transformers v5+ is required.
```bash
pip install transformers>=5.0.0
```

---

## Core Training Parameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `learning_rate` | 2e-4 (SFT), 5e-5 (DPO), 5e-6 (GRPO) | Lower for RL stages |
| `num_train_epochs` | 3 (SFT), 2 (DPO/GRPO) | 1–3 typical |
| `warmup_steps` | 5–50 | 5 for small datasets |
| `lr_scheduler_type` | `"cosine"` | cosine decay standard |
| `optim` | `"adamw_8bit"` | memory efficient |
| `weight_decay` | 0.01 | regularization |
| `seed` | 3407 | Unsloth default |
| `fp16` | false | use bf16 instead |
| `bf16` | true | modern GPUs |
| `save_steps` | 100 | critical for spot instances |
| `resume_from_checkpoint` | true | spot instance safety |
| `gradient_checkpointing` | `"unsloth"` | Unsloth's optimized version |

---

## Method × VRAM Table

| Method | Min VRAM | Recommended | Notes |
|--------|----------|-------------|-------|
| QLoRA (4-bit) | 6GB | 8–16GB | Not recommended for Qwen3.5 |
| LoRA 16-bit | 8GB | 16–24GB | Preferred for Qwen3.5 |
| Full Fine-tune | 40GB+ | 80GB | Requires A100/H100 |
| DPO (LoRA) | 12GB | 24GB | Load SFT checkpoint first |
| GRPO (LoRA) | 16GB | 24–40GB | Needs 8+ completions per prompt |

---

## Model VRAM Requirements (4-bit quantization unless noted)

| Model | HF ID | VRAM (4-bit) | VRAM (LoRA 16-bit) | Context |
|-------|-------|-------------|-------------------|---------|
| Qwen3.5-9B | `unsloth/Qwen3.5-9B` | 6–8 GB | 10–12 GB | 262K |
| Qwen3.5-27B | `unsloth/Qwen3.5-27B` | 16–20 GB | 30–35 GB | 262K |
| Qwen3.5-35B-A3B (MoE) | `unsloth/Qwen3.5-35B-A3B-GGUF` | 8–10 GB | N/A | 262K |
| Gemma 4 E4B (MoE) | `unsloth/gemma-4-E4B` | 6–8 GB | 10 GB | 128K |
| Gemma 4 27B | `unsloth/gemma-4-27b-it` | 16 GB | 30 GB | 128K |
| Qwen3-8B | `unsloth/Qwen3-8B-bnb-4bit` | 6 GB | 10 GB | 128K |
| Llama 3.1 8B (legacy) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | 6 GB | 10 GB | 128K |

**Tip**: Check `https://huggingface.co/collections/unsloth` for newer models.

---

## LoRA Hyperparameters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                # LoRA rank — higher = more capacity, more VRAM
    lora_alpha=32,       # lora_alpha = r * 2 (standard rule)
    lora_dropout=0.0,    # 0.0 recommended (Unsloth guidance)
    target_modules=[     # all attention + FFN layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,    # True = rank-stabilized LoRA (experimental)
)
```

---

## GRPO Config (RL-specific params)

```python
GRPOConfig(
    num_generations=8,      # completions per prompt — min 8 recommended
    epsilon=0.2,            # PPO clip ratio (standard)
    epsilon_high=0.28,      # high epsilon for exploration
    loss_type="grpo",       # use "grpo" not "reinforce"
    temperature=0.7,        # generation temperature
    max_new_tokens=512,     # max completion length
    # min 300 training steps for GRPO to converge
)
```

**GRPO minimum steps**: 300 steps minimum before evaluating quality. GRPO converges slower than SFT.

---

## Model-Specific Gotchas

### Qwen3.5
- **Requires Transformers v5+**: `pip install transformers>=5.0.0`
- **QLoRA NOT recommended**: Use LoRA 16-bit instead (Unsloth guidance)
- **201 languages**: Best choice for multilingual medical data
- **262K context**: Ideal for long-context EHR tasks
- Enable thinking mode with `enable_thinking=True` in generation

### Gemma 4
- **use_cache fix**: If you see cache-related errors, add `model.config.use_cache = True` after loading
- **num_kv_shared_layers**: For Gemma 4 26B/31B, set `num_kv_shared_layers` in config
- Excellent instruction following out of the box
- Best for patient Q&A and consumer-facing applications

### General
- Always use `bf16=True, fp16=False` on modern GPUs (A100, H100, RTX 30/40 series)
- `use_gradient_checkpointing="unsloth"` is faster than standard `True`
- Chat template must match model: `get_chat_template(tokenizer, chat_template="chatml")`

---

## Export Commands

```python
# Save LoRA adapter only (smallest, remerge later)
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# Save merged 16-bit (for Ollama / full inference)
model.save_pretrained_merged("./merged_16bit", tokenizer, save_method="merged_16bit")

# GGUF export (for llama.cpp / Ollama / LM Studio)
model.save_pretrained_gguf("./gguf_q4", tokenizer, quantization_method="q4_k_m")

# Multiple GGUF sizes at once
model.push_to_hub_gguf(
    "your-hf-username/model-name",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "f16"],
    token="hf_...",
)

# Push LoRA adapter to HuggingFace Hub
model.push_to_hub("your-hf-username/model-name", token="hf_...")
tokenizer.push_to_hub("your-hf-username/model-name", token="hf_...")

# Merge LoRA adapter into base and push merged model to HuggingFace Hub
# Use this when base model = LoRA repo (Unsloth Studio error: "already added LoRA adapters")
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "your-hf-username/your-model-lora",
    max_seq_length = 2048,
    load_in_4bit = False,  # must be False for merging
)

model.push_to_hub_merged(
    "your-hf-username/your-model-merged",
    tokenizer,
    token = "hf_...",
)
```

**GGUF quantization choice guide**:
| Method | Size | Quality | Use case |
|--------|------|---------|---------|
| `q4_k_m` | ~4GB (7B) | Good | Default recommendation — balance of size/quality |
| `q8_0` | ~7GB (7B) | Better | Near-lossless, fits in 12GB VRAM |
| `f16` | ~14GB (7B) | Best | Full precision, inference on large GPU |

---

## Loss Interpretation

| Loss | Interpretation | Action |
|------|----------------|--------|
| 1.5–2.0 | Normal early training | Continue |
| 0.8–1.2 | Good convergence | Continue |
| 0.5–0.8 | Learning well | Monitor eval loss |
| 0.2–0.5 | Possible overfitting | Check eval loss, reduce epochs |
| < 0.2 | Likely overfitting | Stop; reduce LR or epochs |
| Flat | Not learning | Increase LR; check data format |

---

## Useful Commands

```bash
# Check GPU VRAM
nvidia-smi

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check installed Unsloth version
python -c "import unsloth; print(unsloth.__version__)"

# Verify CUDA / torch compatibility
python -c "import torch; print(f'CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}')"

# HuggingFace login (for gated datasets)
huggingface-cli login
```
