# AGENTS.md - MedQA Auto-Training Data Generation Skills

## Essential Commands
- `/medqa` - Launch interactive wizard for medical LLM training setup
- `/medqa [params]` - Wizard with inline parameters (Use-case, Difficulty, VRAM, Dataset, etc.)
- Outputs generated to `./medqa-training/` directory

## Skill Location
- Main skill: `~/.claude/skills/medqa-dataset-selector/`
- References: `~/.claude/skills/medqa-dataset-selector/references/`
- Generated files: `./medqa-training/` (dataset_selection.md, train_config.yaml, train.py, README.md)

## Verification Steps (Run After `/medqa` Completion)
1. Check `./medqa-training/` contains all 4 required files
2. Validate `train.py` has valid Python syntax with curriculum ordering
3. Confirm `train_config.yaml` has no unresolved placeholders
4. Verify `dataset_selection.md` cites research backing for dataset choices

## Critical Workflow Notes
- **Phase 0 (Optional)**: Synthetic data generation via distillation (triggered by `Generate: yes`)
- **SFT → DPO**: Recommended for complex reasoning tasks (+7–8% clinical reasoning gain)
- **Curriculum ordering**: Easy→hard data sequencing in generated train.py
- **CoT filter**: Reasoning traces <100 tokens filtered for reasoning datasets
- **Official splits**: Always preferred when available (MedQA-USMLE, PubMedQA, MedCaseReasoning)

## Dataset Tiers (Research-Backed)
- **Tier 1**: Hard reasoning (MedCaseReasoning, MedXpertQA, DiagnosisArena, MedS-Ins subsets)
- **Tier 2**: Standard quality (MedQA-USMLE, medical-o1-reasoning-SFT, PubMedQA, MedQuAD, MedMCQA)
- **Tier 3**: Patient-facing (HealthCareMagic-100k, Clinical Camel)

## Troubleshooting
- API keys: See `./medqa-training/generate/README_generate.md`
- distilabel: Handled by generate.sh script
- VRAM mapping: Refer to hardware→config table in skill references