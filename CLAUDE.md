# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This project contains Claude Code **skills** for medical LLM training data generation and dataset selection. Skills are interactive wizards invoked via `/skill-name` commands.

## Skills

### `/medqa` — Medical LLM Dataset Selector + Training Config Generator

Located at: `medqa-dataset-selector/SKILL.md`

An interactive wizard that:
1. (Optional) Generates synthetic training data via `distilabel` from your own documents
2. Guides dataset selection using research-validated medical datasets (Nature/NPJ/ICML 2024–2025)
3. Recommends training method: SFT, SFT→DPO, or SFT→GRPO with curriculum
4. Generates complete, runnable Unsloth training scripts (`train.py`, `train_config.yaml`, `dataset_selection.md`, `README.md`)

**Reference files** (in `medqa-dataset-selector/references/`):
- `dataset-catalog.md` — 4-tier dataset catalog with research citations
- `phase-workflows.md` — VRAM tables, format converters, all 3 train script templates
- `distillation-workflow.md` — distilabel pipeline YAMLs, session model auto-detection
- `unsloth-cheatsheet.md` — Unsloth params, VRAM, model gotchas, export commands
- `runpod-cheatsheet.md` — RunPod GPU pricing, pod setup, spot instance safety

**Output** (written to `./medqa-training/` in CWD):
- `dataset_selection.md` — selected datasets with research backing
- `train_config.yaml` — complete Unsloth config
- `train.py` or `train_sft.py`, `train_dpo.py`, `train_grpo.py`
- `README.md` — setup instructions, eval recommendations

## Key Research Basis

- **MedCaseReasoning** (Stanford arXiv 2025): +29% diagnostic accuracy, +41% reasoning recall
- **DeepSeek-R1** (Nature 2025): GRPO with rule-based rewards (no neural RM)
- **Fleming-R1** (arXiv 2025): Two-stage GRPO — SFT cold start required before GRPO
- **Med-RLVR** (arXiv 2025): RLVR +8% OOD accuracy vs SFT
- **Meerkat** (npj Digital Medicine 2025): CoT from authoritative textbooks → +22.3%
- **DPO vs SFT** (JMIR 2025): DPO after SFT → +7–8% clinical reasoning accuracy

## Important Notes

- MIMIC/PhysioNet datasets require DUA + credentialing; fine-tuned models cannot be publicly released
- Qwen3.5 requires Transformers v5+; QLoRA not recommended (use LoRA 16-bit)
- GRPO requires SFT cold start first — direct GRPO from random init diverges
- Distilabel teacher model = current session model (auto-detected from CLAUDE_MODEL/OPENCODE_MODEL env var)
