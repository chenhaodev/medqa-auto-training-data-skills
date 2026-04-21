---
name: medqa
description: "Use this skill whenever a user wants to fine-tune or train a medical LLM — including USMLE prep, clinical reasoning, patient Q&A, clinical notes, drug info, or diagnosis tasks. Handles the full setup: research-backed dataset selection (MedCaseReasoning, MedQA-USMLE, MedXpertQA, MIMIC, etc.), training method recommendation (SFT, DPO, or GRPO with curriculum), and generating complete runnable Unsloth scripts (train.py, train_config.yaml, dataset_selection.md, README.md). Also triggers when the user asks which medical datasets to use for fine-tuning, wants a GRPO reward function for medical MCQ, needs a distilabel pipeline to generate synthetic medical QA from their own documents, or mentions Unsloth + medical in the same context. Use even if the user does not say 'dataset selection' — any medical LLM training setup request should use this skill."
allowed-tools: AskUserQuestion, Read, Write, WebSearch, Bash
---

# /medqa — Medical LLM Dataset Selector + Training Config Generator

## Overview

This skill guides you through selecting the best research-validated medical datasets and generates a complete, runnable Unsloth training setup. Output goes to `./medqa-training/` in your current working directory.

References live in `references/` alongside this SKILL.md:
- `references/phase-workflows.md` — per-phase logic, VRAM tables, train script templates
- `references/dataset-catalog.md` — curated dataset catalog with research backing
- `references/distillation-workflow.md` — distilabel pipeline for synthetic data generation
- `references/unsloth-cheatsheet.md` — Unsloth params, VRAM, export, gotchas
- `references/runpod-cheatsheet.md` — RunPod GPU pricing, setup, spot best practices

---

## STEP 0 — Parse Inline Parameters (check FIRST)

Scan the `/medqa` invocation arguments for these recognized keys:

| Key | Values |
|-----|--------|
| `Use-case` | usmle-prep, patient-qa, clinical-notes, drug-info, diagnosis, reasoning, custom |
| `Difficulty` | standard, complex, expert, long-context |
| `VRAM` | 8GB, 12GB, 16GB, 24GB, 40GB+ |
| `Dataset` | any HuggingFace dataset ID or shortname |
| `Model` | any model name/HF ID |
| `Method` | sft, dpo, grpo |
| `Format` | alpaca, sharegpt, chatml, custom |
| `Language` | english, multilingual, or specific language |
| `Generate` | yes / no |

**Skip logic**:
- `Generate: yes` → enter Phase 0 before anything else
- All three of (Use-case + Difficulty + VRAM) provided → skip Phases 1–2, go directly to Phase 3
- Use-case + VRAM provided → skip Phase 1, go to Phase 2
- Use-case only → skip Phase 1

---

## PHASE 0 — Synthetic Data Generation (OPTIONAL)

**Trigger**: enter Phase 0 when ANY of these are true:
- Inline param `Generate: yes`
- User mentions their own documents — phrases like "my PDFs", "my textbooks", "our clinical guidelines", "hospital protocols", "proprietary data", "internal documents", "my own data"
- User asks to "generate" or "create" training data from source material they have

If the invocation has no arguments (bare `/medqa`) and no document mentions, skip Phase 0 — don't ask about it unless the user brings it up.

### Auto-detect teacher model (NO user question needed)

```python
import os
session_model = os.environ.get("CLAUDE_MODEL", os.environ.get("OPENCODE_MODEL"))

if session_model and "claude" in session_model:
    teacher_type = "AnthropicLLM"
    teacher_model = session_model
elif session_model:
    teacher_type = "OllamaLLM"
    teacher_model = session_model
else:
    teacher_type = "AnthropicLLM"
    teacher_model = "claude-sonnet-4-6"
```

Display to user: "Teacher model auto-detected: `{teacher_model}` — no additional API key needed."

### Phase 0 Questions (1 batched AskUserQuestion, 3 questions)

1. **Source documents**:
   - A) Local files — provide path (PDF/txt/JSONL)
   - B) HuggingFace dataset — provide HF ID
   - C) Topic-only — no documents (use topic prompting)

2. **Output QA type**:
   - A) Simple factual QA (question + short answer)
   - B) Complex reasoning with CoT traces (for SFT cold start / reasoning training)
   - C) Preference pairs (chosen/rejected for DPO)

3. **Scale**:
   - A) Small (500–2K pairs, ~30–60 min)
   - B) Medium (5K–10K pairs, ~2–4 hours)
   - C) Large (10K+ pairs — show time estimate before proceeding)

### Phase 0 Output

Read `references/distillation-workflow.md` for templates. Generate to `./medqa-training/generate/`:

```
./medqa-training/generate/
├── medical_qa_pipeline.yaml   ← distilabel pipeline config
├── generate.sh                ← one-command runner
└── README_generate.md         ← explains teacher model, expected output
```

After Phase 0 completes, add generated data to the dataset pool for Phase 3 selection.

---

## PHASE 1 — Use-Case & Difficulty Definition

**Skip if**: Use-case provided inline.

### Phase 1 Questions (1 batched AskUserQuestion, 4 questions)

1. **Medical task** (choose one or describe):
   - A) USMLE prep (Step 1–3 MCQ)
   - B) Patient Q&A (conversational, consumer-facing)
   - C) Clinical Notes / EHR summarization
   - D) Drug information
   - E) Diagnosis / differential reasoning
   - F) Complex clinical reasoning (multi-step)
   - G) Custom — describe your task

2. **Difficulty tier** (affects dataset selection significantly):
   - A) Standard exam-style (MCQ, USMLE Step 1–3, broad knowledge)
   - B) Complex clinical reasoning (multi-step differential, clinical case reports from NEJM/Lancet)
   - C) Expert specialist level (ABMS board-level, rare disease, multimodal imaging)
   - D) Long-context planning (multi-note EHR, discharge summaries, care planning)

   > Research note: MedCaseReasoning hard clinical cases → +29% diagnostic accuracy (Stanford 2025). Choosing Complex or Expert difficulty unlocks significantly better training signal than Standard alone.

3. **Conversation style**:
   - A) Single-turn (one question, one answer)
   - B) Multi-turn (dialogue, follow-up questions)
   - C) Structured reasoning with CoT traces (think-step-by-step format)

4. **Language**:
   - A) English only
   - B) Multilingual (recommend Qwen3.5 — 201 languages, 262K context)
   - C) Specific language — which?

---

## PHASE 2 — Hardware & Training Method

**Skip if**: VRAM provided inline.

### Phase 2 Questions (1 batched AskUserQuestion, 3 questions)

1. **GPU VRAM**:
   - A) 8GB (consumer GPU, RTX 3070/4060)
   - B) 12GB (RTX 3080/4070)
   - C) 16GB (RTX 3080Ti/4080/A4000)
   - D) 24GB (RTX 3090/4090/A5000) ← recommended, great price/performance on RunPod
   - E) 40GB+ (A100/H100) ← large models or GRPO

   > See `references/runpod-cheatsheet.md` for GPU rental costs if you don't have local hardware.

2. **Training time budget**:
   - A) Quick (under 2 hours) — subset datasets to ~5K examples
   - B) Standard (2–8 hours) — full recommended dataset size
   - C) Extended (overnight, 8–24 hours) — full datasets + curriculum

3. **Training method** (with research guidance):
   - A) **SFT** — default; good for broad knowledge, patient Q&A, drug info
   - B) **SFT → DPO** — recommended for complex clinical reasoning; +7–8% over SFT alone (JMIR 2025)
   - C) **SFT → GRPO** — best for verifiable MCQ tasks (USMLE, MedMCQA, MedXpertQA); requires answer keys; two-stage: cold-start SFT on CoT traces → GRPO with curriculum (Fleming-R1 / DeepSeek-R1, Nature 2025)

   > GRPO tip: Rule-based rewards only (no neural reward model) — format + accuracy + CoT-length. Neural RMs get exploited at scale (DeepSeek-R1).

4. **Training interface**:
   - A) **Unsloth Studio** (GUI / web UI) — generates a step-by-step UI guide instead of Python scripts
   - B) **CLI / scripts** — generates runnable `train_sft.py`, `train_dpo.py` etc. (default)

---

## PHASE 3 — Dataset Selection

### Step 3a — Freshness & Language Check (do this BEFORE reading catalog)

The dataset catalog is a curated starting point, not a live index. Before recommending from it:

**If Language is NOT English** (Chinese, Japanese, Arabic, etc.):
1. Run a WebSearch: `"{language} medical LLM fine-tuning dataset huggingface 2025 2026"`
2. Check for language-native alternatives to every catalog recommendation — native-language datasets nearly always outperform translated or cross-lingual ones for non-English inference
3. Prefer datasets published in the last 12 months where available
4. Present discovered datasets alongside catalog options, clearly labeling which are native-language

**For any language** (including English):
- Scan catalog entry dates — entries marked older than ~18 months may have been superseded
- If the catalog's top recommendation for a use-case is from 2023 or earlier, do a quick WebSearch to check for newer alternatives: `"medical reasoning dataset huggingface {year}"`
- You don't need to search for every dataset — use judgment. Hard reasoning datasets (Tier 1) change less often; broad instruction-tuning datasets turn over faster.

The goal is to not silently hand the user a stale list. If the catalog is current enough, say so and proceed. If you found better options, surface them.

### Step 3b — Filter and recommend

Read `references/dataset-catalog.md`. Filter by:
- Method (SFT → any tier; DPO → prefer Tier 1+2 with rich reasoning; GRPO → requires answer keys)
- Use-case (clinical notes → Tier 4; patient Q&A → Tier 3; reasoning → Tier 1)
- Difficulty tier (standard → Tier 2; complex/expert → Tier 1; EHR → Tier 4)

Present top 3–5 datasets with research citations. For each, show: HF ID, language, last-verified date, and expected performance gain where available. Flag any English-only dataset being recommended to a non-English user.

**MIMIC gating logic** (mandatory check):
- If clinical notes or EHR use-case → ask: "Do you have PhysioNet credentialing and have signed the MIMIC Data Use Agreement?"
  - Yes → explain DUA + model release restriction (trained model weights cannot be publicly released), then provide MIMIC dataset access instructions
  - No → auto-redirect to `starmpcc/Asclepius-Synthetic-Clinical-Notes` (157K synthetic discharge summaries, no DUA required)

### Phase 3 Questions (1 batched AskUserQuestion, 3 questions)

1. **Dataset selection** (multi-select from presented options, or type your own HF IDs):

2. **Example count**:
   - A) Small (1K–5K) — fast iteration, verify pipeline
   - B) Medium (10K–50K) — balanced quality/time
   - C) Full recommended size (shown per dataset)
   - D) Custom number

3. **Custom data** (optional):
   - Any additional local files or HF datasets to mix in?
   - If yes → what format? (alpaca/sharegpt/chatml/raw)

---

## PHASE 4 — Data Quality & Preprocessing

**Branch on training interface from Phase 2:**

---

### CLI path — Automated pipeline (no user questions)

For CLI output, **never ask the user about format or cleaning** — detect and apply the full pipeline programmatically. The pipeline runs in this fixed order:

#### 1. Auto-detect format from column names

```python
def detect_format(ds):
    cols = set(ds.column_names)
    if "case_presentation" in cols:              return "medcase"
    if "conversations" in cols:                  return "sharegpt"
    if "messages" in cols:                       return "chatml"
    if "instruction" in cols:                    return "alpaca"
    if "question" in cols and "options" in cols: return "medqa"
    raise ValueError(f"Unknown format. Columns: {sorted(cols)}")
```

#### 2. Normalize all formats to `[{role, content}, …]` conversations

- Inject `SYSTEM_PROMPT` describing the training domain into every conversation
- ShareGPT: use `standardize_sharegpt` from `unsloth.chat_templates`
- Alpaca: map `instruction`/`input`/`output` → user/assistant turns
- MedCase: wrap `case_presentation` as user; `<think>{reasoning}</think><answer>{diagnosis}</answer>` as assistant
- ChatML/messages: pass through directly

#### 3. Pool all datasets before selection

Never select from each dataset separately. Merge all normalized conversations into a single pool first, then apply dedup + selection to the unified pool.

#### 4. Dedup by MD5 of the last user turn (cross-dataset)

```python
key = last_user_message.lower().strip()
h = hashlib.md5(key.encode()).hexdigest()
```

#### 5. Quality filter (always applied)

Drop examples where:
- User turn < 20 words
- `<think>` trace is present but < 50 words (degenerate CoT)
- Tokenized length > 90% × `MAX_SEQ_LENGTH` (would be truncated mid-reasoning)

#### 6. Score each example for difficulty

```
score = min(cot_words/100, 3.0)   # CoT length in <think> block
      + 1.0 if has <think> tags    # reasoning format present
      + min(user_words/200, 1.0)   # user turn complexity
```

#### 7. Hard-biased stratified selection

Select `n_target` examples with quartile budget:

| Quartile | Difficulty | Share |
|----------|-----------|-------|
| Q4 | Hardest | 40% |
| Q3 | Hard | 30% |
| Q2 | Medium | 20% |
| Q1 | Easiest | 10% |

Rationale: MedCaseReasoning (14K hard cases) → +29% over equal mixing. Hard-bias is research-validated.

#### 8. Curriculum sort: ascending difficulty

Sort selected pool by score ascending before `apply_chat_template` — easy examples first, hard last. SFT trainer sees easy→hard naturally.

#### 9. Apply chat template

```python
texts = tokenizer.apply_chat_template(pool, tokenize=False)
```

#### GRPO-specific additions

- Filter questions < 20 words and options < 3 chars per option
- Compute actual prompt token lengths; filter to 90th percentile
- Set `max_prompt_length = p90_token_count + 1` (dynamic, not a hardcoded constant)
- Phase B: cross-dataset dedup, then mix 30% USMLE + 70% MedMCQA clinical subjects

MedMCQA clinical subjects (Phase B only — exclude preclinical):
```python
CLINICAL_SUBJECTS = {
    "Medicine", "Surgery", "Gynaecology & Obstetrics", "Pediatrics",
    "Psychiatry", "Ophthalmology", "ENT", "Anaesthesia",
    "Radiology", "Pathology", "Microbiology",
}
```

---

### Studio path — Manual questions (3 questions)

1. **Format conversion**:
   - A) Auto-detect and convert to ChatML (recommended)
   - B) Keep original format
   - C) Custom format — describe target schema

2. **Cleaning steps** (multi-select):
   - A) Deduplication (MD5 on instruction field)
   - B) Length filter (remove <20 tokens, remove >`max_seq_length * 0.9` tokens)
   - C) CoT quality gate (filter reasoning traces < 100 tokens)
   - D) PII scrub (basic pattern matching for names, DOB, MRN)
   - E) None — use raw data

3. **Train/eval split**:
   - A) Use official splits (recommended: MedQA-USMLE, PubMedQA, MedCaseReasoning)
   - B) 90/10 random split
   - C) 80/20 random split

**Notes for Studio path**:
- Curriculum ordering: train easy-tier datasets first, hard-tier second (sequential runs, same principle as CLI)
- Sequential training does not require cross-dataset deduplication
- Official test splits must not be used for training

See `references/phase-workflows.md` for format converter code.

---

## PHASE 5 — Config Generation (automated)

**Branch on training interface selected in Phase 2:**

- **CLI / scripts** → follow standard flow below, generate Python scripts
- **Unsloth Studio** → skip Python scripts entirely; generate `UNSLOTH_STUDIO_GUIDE.md` instead (see "Studio Output" section below)

---

### Studio Output — `UNSLOTH_STUDIO_GUIDE.md`

When the user chose Unsloth Studio, generate a UI field-by-field guide instead of scripts. The guide must cover:

**Key Studio-specific facts** to include at the top:
- Unsloth Studio cannot mix multiple datasets in one run — train one dataset at a time
- When training datasets sequentially (not mixing), **no cross-dataset deduplication is needed**
- After each dataset completes, use **"Merge & Save"** before starting the next — this merges the LoRA adapter into the base weights and creates a full model checkpoint. Do NOT skip this step.
- The next training run should load the *merged* model as its starting point, not the original base model

**How to verify a correct merge**: if the export directory contains BOTH a large `model.safetensors` AND `adapter_config.json`, Unsloth Studio saved both together. The merge happened, but loading this directory as-is will cause Unsloth to re-apply the adapter and error with `RuntimeError: Unsloth: You already added LoRA adapters to your model!`. The fix is to copy the model to a clean directory without the adapter files.

**Sequential training flow** — always show this diagram explicitly:
```
[Base model]
    ↓ Train on Dataset 1 → Merge & Save → [model_v1]
[model_v1]
    ↓ Train on Dataset 2 → Merge & Save → [final_model]
[final_model]
    ↓ Train DPO (if applicable) → Merge & Save → [final_model_dpo]
```

For each training run in the guide, provide a table with every UI field and its value — model path, LoRA r/alpha, batch size, LR, epochs, etc. Don't leave the user guessing which field maps to which parameter.

---

### CLI Output — Standard scripts

#### Step A — Verify against official Unsloth notebooks (do this first)

Before generating any script, fetch the relevant Python script from GitHub to check for API changes that would break generated code:

```bash
# GRPO reference:
curl -s "https://raw.githubusercontent.com/unslothai/notebooks/main/python_scripts/Qwen3_(4B)-GRPO.py"
# SFT reference:
curl -s "https://raw.githubusercontent.com/unslothai/notebooks/main/python_scripts/Qwen3_(14B)-Reasoning-Conversational.py"
```

Cross-check against these known breaking changes from old templates:

| Old pattern (wrong) | Correct (current) |
|---|---|
| `tokenizer=tokenizer` in GRPOTrainer | `processing_class=tokenizer` |
| `load_in_4bit=True` for GRPO | `load_in_4bit=False` (LoRA 16-bit) |
| no `fast_inference` | `fast_inference=True`, `max_lora_rank=LORA_R`, `gpu_memory_utilization=0.9` |
| no `vllm_sampling_params` | `SamplingParams(...)` in GRPOConfig |
| `warmup_steps=5` in GRPO | `warmup_ratio=0.1` |
| `epsilon=0.2, epsilon_high=0.28` | removed (old API) |
| `lr_scheduler_type="cosine"` | `"linear"` |
| `weight_decay=0.01` | `0.001` |
| completion as string | `completion[0]["content"]` (TRL 0.22.2 format) |
| kwarg `answers` | `answer` (must match dataset column name exactly) |
| no `os.environ["UNSLOTH_VLLM_STANDBY"]` | `os.environ["UNSLOTH_VLLM_STANDBY"] = "1"` at top |

**Mandatory version pins** (add to every README):
```bash
pip install unsloth vllm              # vllm required for GRPO fast_inference
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
```

#### Step B — Always generate `train_utils.py` first

`train_utils.py` is a required shared module. Generate it before any training script. It must contain exactly these four components:

**1. `SmoothedLossCallback(beta=0.98, log_dir)`**
- Fires on `on_log` — watches: `loss`, `eval_loss`, `reward`, `rewards/accuracy_reward`, `rewards/match_format_reward`, `rewards/cot_length_reward`, `kl`, `completion_length`
- EMA with bias correction: `ema = beta*ema + (1-beta)*raw` → `corrected = ema / (1 - beta^step)`
- Console display per step: `[  100] loss 1.8423→1.7234 | eval_loss 1.9234→1.9012`
- Appends to `{log_dir}/loss_history.jsonl` for offline plotting
- `on_train_end`: prints summary table (final + best per metric, with ↓/↑ direction)

**2. `BestRewardCallback(patience)` — GRPO only**
- Watches `reward` in logs
- On improvement: prints `★ New best reward = X.XXXX at step N`, sets `control.should_save = True`

**3. `add_export_args(parser)`**
Adds these flags to any argparser:
```
--hub-id STR        HF repo prefix (e.g. "username/risk-model")
--push-lora         Upload LoRA adapter to {hub-id}-lora
--push-merged       Upload merged 16-bit to {hub-id}-merged
--push-gguf         Upload GGUF to {hub-id}-gguf (also saves locally)
--gguf-methods STR  Comma-separated quant methods (default: q4_k_m)
--gguf-local        Save GGUF locally without hub upload
--hf-token STR      HF token (or HF_TOKEN env var)
--save-total INT    Max checkpoints to keep (default: 3)
```

**4. `export_model(model, tokenizer, output_dir, args)`**
- Always saves merged 16-bit locally first
- `--push-lora` → `model.push_to_hub("{hub-id}-lora", token=token)`
- `--push-merged` → `model.push_to_hub_merged("{hub-id}-merged", ...)`
- `--push-gguf` → `model.push_to_hub_gguf("{hub-id}-gguf", tokenizer, quantization_method=[...], token=token)`
- Warns clearly if `--push-*` set without `--hub-id` or token

#### Step C — Generate `train_sft.py`

Must embed the full Phase 4 data pipeline as functions (`detect_format`, `normalize_to_conversations`, `dedup`, `quality_filter`, `score_difficulty`, `hard_biased_select`). Never hardcode per-dataset loading — the pipeline is generic.

SFT trainer config additions vs old templates:
```python
callbacks=[SmoothedLossCallback(log_dir=out_dir)],
args=SFTConfig(
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=args.save_total,
    load_best_model_at_end=True,      # disabled by --no-best
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    padding_free=False,               # set True if >17GB VRAM free after model load
    report_to="none",
)
```

Replace final `save_pretrained_merged()` call with `export_model(model, tokenizer, args.output, args)`.

#### Step D — Generate `train_grpo.py`

Must embed Phase B mixing (30% USMLE + 70% MedMCQA clinical subjects) with cross-dataset dedup and dynamic `max_prompt_length` from p90 tokenized lengths.

GRPO trainer additions:
```python
callbacks=[SmoothedLossCallback(log_dir=out_dir), BestRewardCallback()],
args=GRPOConfig(
    save_total_limit=args.save_total,
    ...
)
```

Use 3 separate reward functions — never one combined function:
`match_format_reward`, `accuracy_reward`, `cot_length_reward` → passed as list to `reward_funcs=[...]`.

Replace final save with `export_model(model, tokenizer, args.output, args)`.

---

Read `references/phase-workflows.md` for templates. Auto-select model using:

| Use-case | Method | VRAM | Model |
|----------|--------|------|-------|
| USMLE/reasoning | SFT | ≤12GB | `unsloth/Qwen3.5-9B` |
| USMLE/reasoning | SFT or GRPO | 16–24GB | `unsloth/Qwen3.5-27B` |
| Hard reasoning/GRPO | SFT→GRPO | any | `unsloth/Qwen3.5-9B` (cold start) |
| Patient Q&A | SFT | ≤12GB | `unsloth/gemma-4-E4B` |
| Patient Q&A | SFT | 16GB+ | `unsloth/gemma-4-27b-it` |
| Multilingual | SFT | any | `unsloth/Qwen3.5-9B` |
| Long-context EHR | SFT | 16GB+ | `unsloth/Qwen3.5-27B` (262K context) |
| Clinical notes / DPO | SFT→DPO | 24GB+ | `unsloth/Qwen3.5-27B` |

Apply VRAM → config mapping from `references/phase-workflows.md`.

**Generate output files** to `./medqa-training/`:

```
./medqa-training/
├── dataset_selection.md    ← selected datasets with research citations
├── train_config.yaml       ← complete Unsloth config (no placeholder tokens)
├── train.py / train_sft.py ← SFT script with curriculum ordering
├── train_dpo.py            ← (DPO method only)
├── train_grpo.py           ← (GRPO method only; includes medical_reward_fn)
└── README.md               ← setup instructions, eval recommendations
```

### Syntax-check all generated Python scripts

After writing every `.py` file, run:
```bash
python -m py_compile ./medqa-training/train_utils.py 2>&1 || true
python -m py_compile ./medqa-training/train_sft.py 2>&1 || true
python -m py_compile ./medqa-training/train_dpo.py 2>&1 || true
python -m py_compile ./medqa-training/train_grpo.py 2>&1 || true
```

Fix any parse error before proceeding — template substitution mistakes surface here before the user hits them at runtime.

### Final confirmation (1 AskUserQuestion)

"Config generated! What would you like to do next?"
- A) Looks good — I'll run it now
- B) Adjust config — which parameter?
- C) Explain choices — which decision?
- D) Done

---

## Output Spec

### `dataset_selection.md`
- For each selected dataset: name, HF ID, size used, research citation with expected gain
- Combination rationale
- License notes (highlight any DUA/restricted datasets)

### `train_config.yaml`
- All required Unsloth params filled in
- No `{placeholder}` tokens remaining
- Commented with research rationale for key choices

### `train_utils.py` (CLI method — always generated first)
- `SmoothedLossCallback`: EMA-smoothed live display + JSONL log + end-of-training summary
- `BestRewardCallback`: GRPO best-reward tracking with checkpoint trigger
- `add_export_args(parser)`: shared `--hub-id`, `--push-lora/merged/gguf`, `--gguf-methods`, `--hf-token`, `--save-total` flags
- `export_model(model, tokenizer, output_dir, args)`: local merged save + conditional HF push

### `train.py` / `train_sft.py`
- Full Phase 4 data pipeline embedded: `detect_format` → `normalize_to_conversations` → `dedup` → `quality_filter` → `score_difficulty` → `hard_biased_select` → curriculum sort → `apply_chat_template`
- Best-checkpoint saving: `load_best_model_at_end=True`, `evaluation_strategy="steps"`, `save_total_limit`
- `SmoothedLossCallback` in `callbacks=`
- Ends with `export_model()` instead of bare `save_pretrained_merged()`

### `train_grpo.py` (GRPO method)
- Phase A: MedQA-USMLE with quality filter + complexity-scored subset
- Phase B: 30% USMLE + 70% MedMCQA clinical subjects, cross-dataset dedup, dynamic `max_prompt_length` from p90
- Three separate reward functions: `match_format_reward`, `accuracy_reward`, `cot_length_reward`
- `SmoothedLossCallback` + `BestRewardCallback` in `callbacks=`
- `os.environ["UNSLOTH_VLLM_STANDBY"] = "1"` at module top
- `load_in_4bit=False`, `fast_inference=True`, `max_lora_rank`, `gpu_memory_utilization=0.9`
- Ends with `export_model()`

### `README.md`
- Prerequisites: `pip install unsloth vllm`, `transformers==4.56.2`, `--no-deps trl==0.22.2`
- HuggingFace login instructions (for gated datasets)
- PhysioNet warning if MIMIC datasets included
- Export/upload CLI reference table
- RunPod quick-start link to `references/runpod-cheatsheet.md`
- Eval recommendation: run `openai/healthbench` after training

---

## Error Recovery

- **HuggingFace dataset not found**: Check ID spelling, suggest alternative from catalog
- **VRAM mismatch**: Suggest smaller model or increased quantization
- **MIMIC without credentialing**: Auto-redirect to Asclepius-Synthetic
- **GRPO without answer keys**: Warn user that GRPO requires verifiable answer keys; suggest SFT→DPO instead
- **distilabel not installed**: Show `pip install distilabel` command in generate.sh
- **Transformers version (Qwen3.5)**: Qwen3.5 requires Transformers v5+; add version check in README
- **`RuntimeError: Unsloth: You already added LoRA adapters`**: model dir has `adapter_config.json` alongside weights — copy to clean dir and remove adapter files before loading as base
- **`processing_class` not accepted**: TRL version < 0.22.2; upgrade with `pip install --no-deps trl==0.22.2`
- **`fast_inference` not accepted**: Unsloth version too old; re-install with `pip install unsloth vllm`
- **GRPO OOM with `num_generations=4`**: Reduce to 2, or reduce `max_completion_length`
- **Unknown dataset format**: `detect_format()` raises `ValueError` with column list — inspect the dataset schema and add a custom normalizer branch
- **HF upload fails without token**: Check `HF_TOKEN` env var or pass `--hf-token`
