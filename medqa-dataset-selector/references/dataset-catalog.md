# Medical Dataset Catalog

A curated **starting point** for medical LLM fine-tuning — not an exhaustive or always-current list. Use as a seed: verify freshness and search for language-native alternatives before finalizing recommendations. HuggingFace moves fast and better datasets appear regularly.

Last catalog review: April 2026. Each entry has a `Last verified` date — treat entries older than ~18 months as leads to investigate, not prescriptions to follow blindly.

---

## Tier 1 — Hard Reasoning / Complex Clinical Cases

Highest training signal. Small but potent — research shows these outperform 10× larger noisy datasets.

### MedCaseReasoning
- **HF ID**: `zou-lab/MedCaseReasoning`
- **Size**: 14K train / 2K test (official split)
- **Best for**: Diagnostic reasoning, CoT training, SFT cold start for GRPO
- **Language**: English
- **Avg length**: ~600 tokens per example
- **Research backing**: Stanford arXiv 2025 — +29% diagnostic accuracy, +41% reasoning recall over baseline
- **Format**: Clinical case narrative + step-by-step diagnosis with differential reasoning
- **License**: CC BY 4.0 (open)
- **Last verified**: April 2026
- **Notes**: Use official test split for fair evaluation. Apply CoT quality gate (filter traces < 100 tokens). No Chinese equivalent known as of April 2026 — English reasoning signal still valuable for cross-lingual transfer.

### MedXpertQA
- **HF ID**: `TsinghuaC3I/MedXpertQA`
- **Size**: 4,460 questions (Text subset for training; MM subset for multimodal)
- **Best for**: Expert-level QA, ABMS board-level knowledge, hard eval benchmark
- **Research backing**: ICML 2025 — harder than standard USMLE benchmarks; spans 17 ABMS specialties
- **Format**: MCQ with 5 options + expert explanation
- **License**: CC BY-NC 4.0
- **Notes**: Also serves as held-out eval for hard reasoning. Use Text subset for training; MM subset requires multimodal model.

### DiagnosisArena
- **HF ID**: `shzyk/DiagnosisArena`
- **Size**: 915 cases
- **Best for**: Complex differential diagnosis, multi-specialty reasoning
- **Research backing**: arXiv 2025 — derived from NEJM/Lancet/JAMA case reports
- **Format**: Case presentation → differential diagnosis list → final diagnosis
- **License**: CC BY 4.0
- **Notes**: Best used as hard eval benchmark rather than sole training set. Combine with MedCaseReasoning for training.

### MedS-Ins
- **HF ID**: `Henrychur/MedS-Ins`
- **Size**: 5M instances across 58 corpora / 122 tasks
- **Best for**: Comprehensive instruction tuning, task diversity, broad medical coverage
- **Research backing**: npj Digital Medicine 2024 — MMedIns-Llama 3 trained on this outperformed all prior open medical LLMs
- **Format**: Instruction-following (diverse formats per corpus)
- **License**: Mixed (check per-corpus licenses in dataset card)
- **Notes**: Use subsets of 50K–200K for practical training. Full 5M is dataset-soup — sample strategically. Key subsets: MedQA (USMLE), PubMedQA, medical textbook QA.

---

## Tier 2 — Standard Quality (Research-Validated Classics)

Well-studied, high-quality, commonly used as baselines.

### MedQA-USMLE
- **HF ID**: `GBaker/MedQA-USMLE-4-options`
- **Size**: 10,178 train / 1,273 dev / 1,273 test (official 3-way split)
- **Best for**: USMLE MCQ training, GRPO curriculum Phase A (easiest), standard baseline
- **Research backing**: Gold standard for medical MC — used in Meerkat, Fleming-R1, Med-RLVR, and nearly every medical LLM paper
- **Format**: MCQ with 4 options + answer key
- **License**: Apache 2.0
- **Notes**: ALWAYS use official test split as holdout. For GRPO, use only train split (2,140 questions in GRPO-friendly subsets).

### medical-o1-reasoning-SFT
- **HF ID**: `FreedomIntelligence/medical-o1-reasoning-SFT`
- **Size**: 50K examples
- **Best for**: CoT reasoning training, SFT cold start for GRPO, reasoning trace quality
- **Language**: English
- **Avg length**: ~800 tokens (Meerkat-style long reasoning)
- **Research backing**: Meerkat (npj Digital Medicine 2025) — chain-of-thought from authoritative sources outperformed counterparts by 22.3% on 6 exam datasets
- **Format**: Question + long-form CoT reasoning trace + final answer
- **License**: Apache 2.0
- **Last verified**: April 2026
- **Notes**: Apply CoT quality gate before SFT cold start. Excellent for generating `<think>...</think>` patterns. **Chinese users**: search HuggingFace for `FreedomIntelligence/Medical-R1-Distill-Data-Chinese` or similar — the same team has released Chinese versions; verify current availability.

### PubMedQA
- **HF ID**: `qiaojin/PubMedQA`
- **Size**: 211K total (use `pqa_labeled` subset: 1K gold + 61K silver)
- **Best for**: Evidence-based QA, research paper comprehension
- **Research backing**: Well-established benchmark in medical NLP since 2019; included in MedS-Ins
- **Format**: Research abstract + yes/no/maybe question + reasoning context
- **License**: MIT
- **Notes**: Use `pqa_labeled` (1K gold) for high-quality training. `pqa_unlabeled` (211K) is automatically labeled — lower quality. Official test split available.

### MedQuAD
- **HF ID**: `lavita/MedQuAD`
- **Size**: 47,457 QA pairs
- **Best for**: Drug information, disease Q&A, NIH-sourced factual knowledge
- **Research backing**: Derived from 12 NIH websites — authoritative sourcing, good for factual accuracy
- **Format**: Question + answer pairs (single-turn)
- **License**: CC BY 4.0
- **Notes**: Excellent for drug info and disease facts. Pair with MedCaseReasoning for reasoning tasks.

### MedMCQA
- **HF ID**: `openlifescienceai/medmcqa`
- **Size**: 182,822 train / 4,183 validation / 6,150 test
- **Best for**: Broad medical MCQ, GRPO curriculum Phase B (medium difficulty), topic metadata
- **Research backing**: Used in Fleming-R1 GRPO curriculum; rich subject metadata (anatomy, physiology, etc.)
- **Format**: MCQ with 4 options + answer key + topic
- **License**: Apache 2.0
- **Notes**: Indian medical curriculum — verify alignment with your target exam. Good for GRPO Phase B (6,748 GRPO-subset questions).

---

## Tier 3 — Patient-Facing / Conversational

Lower reasoning density but high volume. Good for patient Q&A fine-tuning.

### HealthCareMagic-100k (ChatDoctor)
- **HF ID**: `lavita/ChatDoctor-HealthCareMagic-iCliniq`
- **Size**: ~100K QA pairs
- **Best for**: Patient Q&A, consumer-facing medical chatbot, conversational style
- **Caveats**: Inconsistent answer quality; potential PII; answers from online doctor-patient forums
- **Format**: Patient question + doctor response
- **License**: CC BY-NC 4.0
- **Notes**: Apply PII scrub and length filter. Validate a random sample before use. Do NOT use for high-stakes clinical reasoning tasks.

### Clinical Camel
- **HF ID**: `wanglab/ClinicalCamel-40K`
- **Size**: 40K QA pairs
- **Best for**: Multi-turn medical Q&A, GPT-4 quality conversations
- **Caveats**: GPT-4 generated — not clinician-verified; may contain GPT hallucinations
- **Format**: Multi-turn dialogue (ShareGPT format)
- **License**: CC BY-NC 4.0
- **Notes**: Higher quality than HealthCareMagic but not peer-reviewed. Good for conversational style tuning.

---

## Tier 4 — Clinical Notes / EHR

### MIMIC-IV Extended (GATED — PhysioNet DUA Required)

⚠️ **Access requirements**: PhysioNet credentialing + signed Data Use Agreement. **Fine-tuned model weights trained on MIMIC cannot be publicly released.**

| Dataset | PhysioNet URL | Size | Best For |
|---------|--------------|------|----------|
| MIMIC-IV-Ext-Instr | physionet.org/content/mimic-iv-ext-instr/1.0.0/ | 450K | EHR instruction following (GPT-3.5 generated) |
| MIMIC-IV-Ext-BHC | physionet.org/content/mimic-iv-ext-bhc/ | ~100K | Hospital course summarization (labeled) |
| MIMIC-IV-Ext-CDM | physionet.org/content/mimic-iv-ext-cdm/ | ~50K | Clinical decision making, abdominal pathology |

**Skill behavior**: Always ask user about PhysioNet credentialing before recommending MIMIC datasets.

### Asclepius-Synthetic-Clinical-Notes (PUBLIC — No DUA)
- **HF ID**: `starmpcc/Asclepius-Synthetic-Clinical-Notes`
- **Size**: 157K synthetic discharge summaries
- **Best for**: Clinical notes training without DUA, publicly releasable models
- **Research backing**: npj Digital Medicine 2025 — synthesized from PMC-Patients case reports; clinically realistic
- **Format**: Synthetic discharge summary + clinical QA pairs
- **License**: CC BY 4.0 (open, no DUA)
- **Notes**: Recommended when user lacks PhysioNet credentialing. Generated from public PMC case reports, not real patient data.

### AGBonnet Augmented Clinical Notes (PUBLIC)
- **HF ID**: `AGBonnet/augmented-clinical-notes`
- **Size**: ~30K augmented EHR notes
- **Best for**: Supplementing Asclepius, multi-note reasoning
- **License**: Apache 2.0
- **Notes**: Publicly available, no restrictions on model release.

---

## Evaluation Benchmarks (NOT for training)

Use these for model evaluation after training — do not include in training data.

| Benchmark | HF ID | Size | What It Measures |
|-----------|-------|------|-----------------|
| HealthBench | `openai/healthbench` | 5,000 cases | Complex health conversations graded by 262 physicians across 26 specialties. **Best overall eval.** |
| MedS-Bench | `Henrychur/MedS-Bench` | 11 tasks | Clinical task diversity |
| MedXpertQA (eval) | `TsinghuaC3I/MedXpertQA` | 4,460 | Expert-level specialty board questions, ABMS-level difficulty |
| DiagnosisArena | `shzyk/DiagnosisArena` | 915 | Complex differential diagnosis (held-out benchmark) |

---

## Research-Backed Combinations

> These combinations were validated as of April 2026. For non-English use cases, replace English datasets with language-native equivalents found via WebSearch before adopting these combinations.

| Goal | Primary Dataset | Secondary | Method | Expected Gain |
|------|----------------|-----------|--------|---------------|
| Hard clinical reasoning | `zou-lab/MedCaseReasoning` | `FreedomIntelligence/medical-o1-reasoning-SFT` | SFT → DPO | +29% diagnostic accuracy (Stanford 2025) |
| USMLE / board exam | `GBaker/MedQA-USMLE-4-options` | `TsinghuaC3I/MedXpertQA` | SFT or GRPO | Matches/exceeds physician-level performance |
| Expert specialist | `TsinghuaC3I/MedXpertQA` | `shzyk/DiagnosisArena` (as eval) | SFT → DPO | ABMS-level knowledge |
| Comprehensive coverage | `Henrychur/MedS-Ins` (50K subset) | `lavita/MedQuAD` | SFT | Best task diversity (122 clinical tasks) |
| Patient Q&A | `lavita/ChatDoctor-HealthCareMagic-iCliniq` | `lavita/MedQuAD` | SFT | Broad conversational coverage |
| Long-context planning | `zou-lab/MedCaseReasoning` (long cases) | `AGBonnet/augmented-clinical-notes` | SFT | Multi-note EHR reasoning |
| Clinical notes (public) | `starmpcc/Asclepius-Synthetic-Clinical-Notes` | `lavita/MedQuAD` | SFT | No DUA; 157K synthetic notes |
| Clinical notes (credentialed) | `MIMIC-IV-Ext-Instr` | `MIMIC-IV-Ext-BHC` | SFT | 450K EHR instructions; requires DUA ⚠️ |
| GRPO MCQ curriculum | `GBaker/MedQA-USMLE-4-options` (Phase A) → `openlifescienceai/medmcqa` (Phase B) → `TsinghuaC3I/MedXpertQA` (Phase C) | — | SFT → GRPO | +8% OOD accuracy (Med-RLVR 2025) |

---

## License Summary

| Dataset | License | Restrictions |
|---------|---------|-------------|
| MedCaseReasoning | CC BY 4.0 | None |
| MedXpertQA | CC BY-NC 4.0 | No commercial use |
| DiagnosisArena | CC BY 4.0 | None |
| MedS-Ins | Mixed | Check per-corpus |
| MedQA-USMLE | Apache 2.0 | None |
| medical-o1-reasoning-SFT | Apache 2.0 | None |
| PubMedQA | MIT | None |
| MedQuAD | CC BY 4.0 | None |
| MedMCQA | Apache 2.0 | None |
| HealthCareMagic | CC BY-NC 4.0 | No commercial use |
| Clinical Camel | CC BY-NC 4.0 | No commercial use |
| MIMIC-IV-Ext-* | PhysioNet DUA | Credentialing required; no public model release |
| Asclepius-Synthetic | CC BY 4.0 | None |
| HealthBench (eval) | MIT | Eval only |
