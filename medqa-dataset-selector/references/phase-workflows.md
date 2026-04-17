# Phase Workflows — Templates and Config Tables

Reference for SKILL.md Phase 2, 4, and 5 logic.

---

## VRAM → Training Config Mapping

| VRAM | load_in_4bit | max_seq_length | per_device_batch | lora_r | lora_alpha | grad_accum |
|------|-------------|----------------|-----------------|--------|-----------|-----------|
| 8GB  | true  | 1024 | 1 | 8  | 16  | 8  |
| 12GB | true  | 2048 | 2 | 16 | 32  | 4  |
| 16GB | true  | 2048 | 2 | 16 | 32  | 4  |
| 20GB+| true  | 4096 | 4 | 32 | 64  | 4  |
| 24GB | true  | 4096 | 4 | 32 | 64  | 4  |
| 40GB+| false | 4096 | 8 | 64 | 128 | 2  |

**Rule**: `lora_alpha = lora_r * 2` (standard ratio).
**Rule**: For Qwen3.5 on 8–16GB, LoRA 16-bit is preferred over QLoRA (Unsloth guidance: QLoRA not recommended for Qwen3.5).
**16GB boundary**: Use `Qwen3.5-9B` (not 27B) for exactly 16GB — the 27B model needs 16–20GB at 4-bit and will OOM on a tight 16GB card.

---

## Use-Case + Method → Recommended Model

| Use-case | Method | VRAM | Model HF ID |
|----------|--------|------|------------|
| USMLE/reasoning | SFT | ≤16GB | `unsloth/Qwen3.5-9B` |
| USMLE/reasoning | SFT or GRPO | 20–24GB | `unsloth/Qwen3.5-27B` |
| Hard reasoning/GRPO | SFT→GRPO | 8–16GB | `unsloth/Qwen3.5-9B` (Phase A cold start) |
| Patient Q&A | SFT | ≤16GB | `unsloth/gemma-4-E4B` |
| Patient Q&A | SFT | 20GB+ | `unsloth/gemma-4-27b-it` |
| Multilingual | SFT | any | `unsloth/Qwen3.5-9B` (201 langs, 262K ctx) |
| Long-context EHR | SFT | 20GB+ | `unsloth/Qwen3.5-27B` (262K native context) |
| Clinical notes / DPO | SFT→DPO | 24GB+ | `unsloth/Qwen3.5-27B` |
| MoE (VRAM efficient) | SFT | 8–10GB | `unsloth/Qwen3.5-35B-A3B-GGUF` (3B active) |

---

## Training Time Estimates (RTX 4090, 24GB)

| Examples | Method | Approx Time |
|----------|--------|-------------|
| 5K | SFT | ~45 min |
| 10K | SFT | ~1.5 hr |
| 50K | SFT | ~6 hr |
| 10K | DPO | ~2 hr |
| 14K | GRPO Phase A | ~3 hr |
| 20K | GRPO Phase A+B | ~5 hr |
| 35K | GRPO Full A+B+C | ~8–10 hr |

---

## Format Converters (Python)

```python
def alpaca_to_chatml(example):
    """Convert alpaca format to ChatML."""
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": example["instruction"]})
    if example.get("input"):
        messages[-1]["content"] += f"\n\nInput: {example['input']}"
    messages.append({"role": "assistant", "content": example["output"]})
    return {"messages": messages}


def sharegpt_to_chatml(example):
    """Convert ShareGPT format to ChatML."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = [
        {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in example["conversations"]
    ]
    return {"messages": messages}


def medqa_to_alpaca(example):
    """Convert MedQA-USMLE to alpaca format."""
    opts = example.get("options", {})
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    instruction = (
        f"Answer the following medical question by selecting the best option.\n\n"
        f"Question: {example['question']}\n\nOptions:\n{options_text}"
    )
    answer_key = example.get("answer_idx", example.get("answer", ""))
    answer_text = opts.get(answer_key, answer_key)
    output = f"The correct answer is {answer_key}. {answer_text}"
    return {"instruction": instruction, "input": "", "output": output}


def medcase_to_chatml(example):
    """Convert MedCaseReasoning to ChatML with CoT format."""
    user_msg = (
        f"You are a clinical expert. Analyze this case and provide your diagnostic reasoning.\n\n"
        f"Case: {example['case_presentation']}"
    )
    if example.get("reasoning"):
        assistant_msg = (
            f"<think>\n{example['reasoning']}\n</think>\n"
            f"<answer>{example['diagnosis']}</answer>"
        )
    else:
        assistant_msg = example.get("diagnosis", "")
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return {"messages": messages}


def dedup_by_instruction(dataset):
    """MD5 deduplication on instruction/question field."""
    import hashlib
    seen = set()
    unique = []
    for ex in dataset:
        key = ex.get("instruction") or ex.get("question") or str(ex.get("messages", ""))
        h = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def length_filter(example, min_tokens=20, max_tokens=None):
    """Filter by approximate token count."""
    text = str(example.get("instruction") or example.get("messages") or "")
    token_estimate = len(text.split())
    if token_estimate < min_tokens:
        return False
    if max_tokens and token_estimate > max_tokens:
        return False
    return True


def cot_quality_filter(example, min_cot_tokens=100):
    """Filter out degenerate CoT traces (< min_cot_tokens words in <think> block)."""
    import re
    text = str(example.get("output") or example.get("messages") or "")
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        cot_len = len(match.group(1).split())
        return cot_len >= min_cot_tokens
    return True  # pass through examples without <think> tags
```

---

## `train_config.yaml` Template

```yaml
# Unsloth Training Configuration
# Generated by /medqa skill

model:
  hf_id: "{MODEL_HF_ID}"
  load_in_4bit: {LOAD_IN_4BIT}
  max_seq_length: {MAX_SEQ_LENGTH}

lora:
  r: {LORA_R}
  alpha: {LORA_ALPHA}
  dropout: 0.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  method: "{METHOD}"  # sft | dpo | grpo
  per_device_train_batch_size: {BATCH_SIZE}
  gradient_accumulation_steps: {GRAD_ACCUM}
  warmup_steps: 5
  num_train_epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  weight_decay: 0.01
  seed: 3407
  fp16: false
  bf16: true
  save_steps: 100
  logging_steps: 10
  output_dir: "./outputs"
  resume_from_checkpoint: true  # safety for spot instances

datasets:
  train:
{DATASET_LIST}
  eval_split: 0.1

export:
  gguf_quantization: "q4_k_m"
  push_to_hub: false
```

---

## `train_sft.py` Template

```python
"""
Medical LLM SFT Training Script
Generated by /medqa skill

Research basis:
- Meerkat (npj Digital Medicine 2025): CoT from authoritative sources -> +22.3%
- MedCaseReasoning (Stanford 2025): +29% diagnostic accuracy, +41% reasoning recall
- Curriculum ordering: easy -> hard data mixing improves convergence
"""

import re
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "{MODEL_HF_ID}"
MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}
LOAD_IN_4BIT = {LOAD_IN_4BIT}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,  # auto-detect
)
model = FastLanguageModel.get_peft_model(
    model,
    r={LORA_R},
    lora_alpha={LORA_ALPHA},
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")


# ── Dataset loading with curriculum ordering (easy -> hard) ───────────────────
def format_messages(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )}


# Load datasets in curriculum order (easy -> hard)
easy_datasets = []
hard_datasets = []

# Example curriculum (replace with actual selected datasets):
# Easy: MedQuAD, HealthCareMagic, MedMCQA
# Hard: MedCaseReasoning, MedXpertQA, medical-o1-reasoning-SFT

{DATASET_LOADING_CODE}

# Curriculum concat: easy first, then hard
all_datasets = easy_datasets + hard_datasets
dataset = concatenate_datasets(all_datasets)
dataset = dataset.map(format_messages, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=3407)


# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=SFTConfig(
        per_device_train_batch_size={BATCH_SIZE},
        gradient_accumulation_steps={GRAD_ACCUM},
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2.0e-4,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        seed=3407,
        save_steps=100,
        logging_steps=10,
        output_dir="./outputs",
        resume_from_checkpoint=True,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    ),
)

trainer.train()

# ── Export ────────────────────────────────────────────────────────────────────
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# GGUF export (for llama.cpp / Ollama)
model.save_pretrained_gguf("./final_model_gguf", tokenizer, quantization_method="q4_k_m")
print("Training complete. Model saved to ./final_model and ./final_model_gguf")
```

---

## `train_dpo.py` Template

```python
"""
Medical LLM DPO Training Script (Stage 2 — run after train_sft.py)
Generated by /medqa skill

Research basis:
- JMIR 2025: DPO after SFT improves clinical reasoning accuracy +7-8% for Llama3/Mistral
- Use SFT-trained checkpoint as starting point, NOT base model
"""

from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

MODEL_PATH = "./final_model"  # path to SFT-trained model
MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit={LOAD_IN_4BIT},
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={LORA_R},
    lora_alpha={LORA_ALPHA},
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# DPO dataset must have: prompt, chosen, rejected fields
{DPO_DATASET_LOADING_CODE}

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # use implicit reference (memory efficient)
    tokenizer=tokenizer,
    train_dataset=dpo_dataset["train"],
    eval_dataset=dpo_dataset["test"],
    args=DPOConfig(
        per_device_train_batch_size={BATCH_SIZE},
        gradient_accumulation_steps={GRAD_ACCUM},
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=5.0e-5,  # lower LR for DPO stage
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        seed=3407,
        save_steps=100,
        logging_steps=10,
        output_dir="./outputs_dpo",
        resume_from_checkpoint=True,
        beta=0.1,  # DPO temperature
        max_length=MAX_SEQ_LENGTH,
    ),
)

trainer.train()
model.save_pretrained("./final_model_dpo")
tokenizer.save_pretrained("./final_model_dpo")
model.save_pretrained_gguf("./final_model_dpo_gguf", tokenizer, quantization_method="q4_k_m")
print("DPO training complete. Model saved to ./final_model_dpo")
```

---

## `train_grpo.py` Template

```python
"""
Medical LLM GRPO Training Script (Two-Stage: SFT cold start + GRPO)
Generated by /medqa skill

Research basis:
- DeepSeek-R1 (Nature 2025): rule-based rewards only (no neural RM); GRPO algorithm
- Fleming-R1 (arXiv 2025): SFT cold start required before GRPO; curriculum learning A->B->C
- Med-RLVR (arXiv 2025): RLVR +8% OOD accuracy vs SFT

GRPO Curriculum:
  Phase A: MedQA-USMLE (2,140 Q) - warm up policy on easiest questions
  Phase B: + MedMCQA  (6,748 Q) - add medium difficulty
  Phase C: + MedXpertQA (4,460 Q) - hard specialist questions last

Stage 1 (Cold Start): Run train_sft.py first on CoT reasoning data.
Stage 2 (GRPO): Run this script starting from SFT checkpoint.
"""

import re
from datasets import load_dataset, concatenate_datasets
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

MODEL_PATH = "./final_model"  # SFT cold-start checkpoint (REQUIRED)
MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}
GRPO_PHASE = "C"  # "A", "B", or "C" — controls curriculum


# ── Reward function (rule-based only — no neural reward model) ────────────────
def medical_reward_fn(completions, answers, **kwargs):
    """
    Rule-based reward for medical MCQ.
    Max total reward: 4.5 per completion.
    Based on DeepSeek-R1 (Nature 2025): verifiable rewards avoid reward hacking.
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        reward = 0.0

        # 1. Format reward (0 or 1.0): must have <think>...</think><answer>X</answer>
        has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
        has_answer = bool(re.search(r'<answer>[A-E]</answer>', completion))
        if has_think and has_answer:
            reward += 1.0

        # 2. Accuracy reward (0 or 2.0): correct answer letter
        match = re.search(r'<answer>([A-E])</answer>', completion)
        if match and match.group(1).upper() == str(answer).upper():
            reward += 2.0

        # 3. Reasoning quality reward (0–1.5): CoT length proxy
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if think_match:
            cot_len = len(think_match.group(1).split())
            if cot_len >= 50:   reward += 0.5  # minimal reasoning
            if cot_len >= 150:  reward += 0.5  # substantial reasoning
            if cot_len >= 300:  reward += 0.5  # thorough reasoning

        rewards.append(reward)

    return rewards


# ── Curriculum dataset loader ─────────────────────────────────────────────────
def load_grpo_curriculum(phase):
    """Load datasets in curriculum order. Phase A=easy, B=medium, C=hard."""
    datasets = []

    # Phase A: MedQA-USMLE (easiest, 2,140 train questions)
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    medqa = medqa.map(lambda x: {
        "prompt": (
            f"Answer the following medical question. Use <think>...</think> for reasoning "
            f"and <answer>X</answer> for your final answer (X = A, B, C, D, or E).\n\n"
            f"Question: {x['question']}\n\n"
            + "\n".join(f"{k}. {v}" for k, v in x.get("options", {}).items())
        ),
        "answer": x.get("answer_idx", x.get("answer", "")),
    })
    datasets.append(medqa)

    if phase in ("B", "C"):
        # Phase B: MedMCQA (medium difficulty, use 6,748 subset)
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")
        medmcqa = medmcqa.shuffle(seed=42).select(range(6748))
        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        medmcqa = medmcqa.map(lambda x: {
            "prompt": (
                f"Answer this medical question using <think>...</think> for reasoning "
                f"and <answer>X</answer> for your answer.\n\n"
                f"Question: {x['question']}\n\n"
                f"A. {x['opa']}\nB. {x['opb']}\nC. {x['opc']}\nD. {x['opd']}"
            ),
            "answer": option_map.get(x.get("cop", 0), "A"),
        })
        datasets.append(medmcqa)

    if phase == "C":
        # Phase C: MedXpertQA (hardest, 4,460 specialist questions)
        medxpert = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="train")
        medxpert = medxpert.map(lambda x: {
            "prompt": (
                f"This is an expert-level medical board question. Use <think>...</think> "
                f"for detailed reasoning and <answer>X</answer> for your final answer.\n\n"
                f"Question: {x['question']}\n\n"
                + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(x.get("options", [])))
            ),
            "answer": x.get("answer", "A"),
        })
        datasets.append(medxpert)

    combined = concatenate_datasets(datasets)
    return combined.train_test_split(test_size=0.05, seed=3407)


# ── Model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit={LOAD_IN_4BIT},
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={LORA_R},
    lora_alpha={LORA_ALPHA},
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

dataset = load_grpo_curriculum(GRPO_PHASE)

# ── GRPO Trainer ──────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[medical_reward_fn],
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=GRPOConfig(
        per_device_train_batch_size={BATCH_SIZE},
        gradient_accumulation_steps={GRAD_ACCUM},
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=5.0e-6,   # lower LR for RL stage
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        seed=3407,
        save_steps=100,
        logging_steps=10,
        output_dir=f"./outputs_grpo_phase{GRPO_PHASE}",
        resume_from_checkpoint=True,
        # GRPO-specific (from Unsloth RL guide)
        num_generations=8,          # completions per prompt (min 8 recommended)
        epsilon=0.2,                # PPO clip ratio
        epsilon_high=0.28,
        max_new_tokens=512,
        temperature=0.7,
    ),
)

trainer.train()
model.save_pretrained(f"./final_model_grpo_{GRPO_PHASE}")
tokenizer.save_pretrained(f"./final_model_grpo_{GRPO_PHASE}")
model.save_pretrained_gguf(f"./final_model_grpo_{GRPO_PHASE}_gguf", tokenizer, quantization_method="q4_k_m")
print(f"GRPO Phase {GRPO_PHASE} complete. Saved to ./final_model_grpo_{GRPO_PHASE}")
```

---

## Curriculum Ordering for SFT (Meerkat / Phi-3 finding)

When mixing datasets in SFT, order easy → hard within the training run:

```python
# Easy datasets (broad knowledge, simple Q&A)
easy = [medquad, healthcaremagic, medmcqa]

# Hard datasets (complex reasoning, long CoT)
hard = [medcase_reasoning, medical_o1_sft, medxpert]

# Concatenate easy FIRST, hard SECOND
from datasets import concatenate_datasets
curriculum_dataset = concatenate_datasets(easy + hard)
```

Research note: Random mixing underperforms curriculum ordering (Meerkat 2025, Phi-3 findings).

---

## Loss Interpretation Guide

| Loss range | Interpretation | Action |
|-----------|----------------|--------|
| 1.5–2.0 | Early training, normal | Continue |
| 0.8–1.2 | Healthy convergence | Continue |
| 0.5–0.8 | Good learning | Monitor |
| 0.2–0.5 | Possible overfitting | Reduce epochs or add data |
| < 0.2 | Likely overfitting | Stop, reduce LR, add regularization |
| Flat/no decrease | Not learning | Increase LR, check data format |
