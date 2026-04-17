---
name: medqa
description: Interactive wizard for selecting medical LLM training datasets and generating complete Unsloth fine-tuning configurations (train.py, train_config.yaml, dataset_selection.md, README.md). Supports SFT, DPO, and GRPO methods with research-backed dataset recommendations. Optionally generates synthetic training data via distilabel.
version: 1.0.0
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

**Trigger**: user has proprietary documents (PDFs, textbooks, clinical guidelines) OR explicitly sets `Generate: yes`.

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

---

## PHASE 3 — Dataset Selection

Read `references/dataset-catalog.md`. Filter by:
- Method (SFT → any tier; DPO → prefer Tier 1+2 with rich reasoning; GRPO → requires answer keys)
- Use-case (clinical notes → Tier 4; patient Q&A → Tier 3; reasoning → Tier 1)
- Difficulty tier (standard → Tier 2; complex/expert → Tier 1; EHR → Tier 4)

Present top 3–5 datasets with research citations. Show expected performance gains where available.

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

### Phase 4 Questions (1 batched AskUserQuestion, 3 questions)

1. **Format conversion**:
   - A) Auto-detect and convert to ChatML (recommended)
   - B) Keep original format
   - C) Custom format — describe target schema

2. **Cleaning steps** (multi-select):
   - A) Deduplication (MD5 on instruction field)
   - B) Length filter (remove <20 tokens, remove >`max_seq_length * 0.9` tokens)
   - C) CoT quality gate (filter reasoning traces < 100 tokens — removes degenerate traces)
   - D) PII scrub (basic pattern matching for names, DOB, MRN)
   - E) None — use raw data

3. **Train/eval split**:
   - A) Use official splits (recommended where available: MedQA-USMLE, PubMedQA, MedCaseReasoning)
   - B) 90/10 random split
   - C) 80/20 random split

**Research-backed preprocessing notes** (display to user):
- Curriculum ordering (Meerkat 2025): if mixing datasets, order easy → hard within trainer
- Official test splits must not be used for training (MedQA-USMLE, MedXpertQA, PubMedQA have official held-out sets)
- For GRPO: CoT quality gate is critical — degenerate cold-start traces corrupt GRPO reward signals

See `references/phase-workflows.md` for format converter code.

---

## PHASE 5 — Config Generation (automated)

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

### `train.py` / `train_sft.py`
- Complete runnable script
- Curriculum ordering (easy → hard dataset mixing)
- `save_steps=100` and `resume_from_checkpoint=True` for spot instance safety

### `train_grpo.py` (GRPO method)
- Two-stage: SFT cold start → GRPO
- `medical_reward_fn` with format + accuracy + CoT-length rewards
- Curriculum dataset loader: Phase A (MedQA-USMLE) → Phase B (+MedMCQA) → Phase C (+MedXpertQA)

### `README.md`
- Prerequisites and install commands
- HuggingFace login instructions (for gated datasets)
- PhysioNet warning if MIMIC datasets included
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
