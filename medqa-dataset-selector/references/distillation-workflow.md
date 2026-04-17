# Distillation Workflow — Synthetic Medical Data Generation

Generate synthetic medical training data from your own documents using `distilabel`.

**Tool choice**: `distilabel` (CLI-native) is the recommended path.
- `easy-dataset` (ConardLi/easy-dataset) is a popular alternative (10K+ stars) but is GUI-only (Next.js web app on port 1717) — no native CLI.
- `EasyDistill` (modelscope/easydistill) is more training-oriented (SFT + ranking + RL KD) — better for white-box KD than data generation.

---

## Teacher Model Auto-Detection

The teacher model is the **current session model** — no additional API key or configuration needed.

```python
import os
from distilabel.models import AnthropicLLM, OllamaLLM

session_model = os.environ.get("CLAUDE_MODEL", os.environ.get("OPENCODE_MODEL"))

if session_model and "claude" in session_model:
    teacher = AnthropicLLM(model=session_model)   # uses existing ANTHROPIC_API_KEY
elif session_model:
    teacher = OllamaLLM(model=session_model)       # local Ollama fallback
else:
    teacher = AnthropicLLM(model="claude-sonnet-4-6")  # safe default
```

- **Claude Code session**: `CLAUDE_MODEL` is set automatically (e.g., `claude-sonnet-4-6`). Uses your existing session credentials.
- **OpenCode session**: `OPENCODE_MODEL` is set automatically to the configured model.
- **Offline/no session**: Falls back to Ollama — run `ollama pull qwen2.5:72b` for a strong local teacher.

---

## distilabel Pipeline YAML Template

### Simple Factual QA

```yaml
# medical_qa_pipeline.yaml
# Run with: distilabel pipeline run --config medical_qa_pipeline.yaml

pipeline:
  name: medical_qa_generation
  description: "Generate medical QA pairs from source documents using session model as teacher"

  steps:
    - name: load_documents
      type: LoadDataFromDisk            # or LoadDataFromHub for HF datasets
      output_mappings:
        text: passage
      runtime_parameters:
        data_path: "{SOURCE_PATH}"      # local file path or HF dataset ID

    - name: chunk_text
      type: SentenceSplitter
      input_mappings:
        text: passage
      runtime_parameters:
        chunk_size: 512
        chunk_overlap: 50

    - name: generate_qa
      type: TextGeneration
      input_mappings:
        system_prompt: system_prompt
        instruction: passage
      llm:
        type: "{TEACHER_LLM_TYPE}"      # AnthropicLLM or OllamaLLM
        model: "{TEACHER_MODEL}"
        generation_kwargs:
          temperature: 0.7
          max_new_tokens: 512
      runtime_parameters:
        system_prompt: |
          You are a medical educator. Given the passage below, generate one clinical question
          and a concise, accurate answer based solely on the passage content.
          
          Format your response as:
          Question: [your question]
          Answer: [your answer]
          
          Focus on clinically relevant facts, drug interactions, diagnostic criteria, or
          treatment guidelines mentioned in the passage.

    - name: score_quality
      type: UltraFeedback
      llm:
        type: "{TEACHER_LLM_TYPE}"
        model: "{TEACHER_MODEL}"

    - name: filter_quality
      type: FilterByScore
      runtime_parameters:
        threshold: 0.7
        field: overall_score

    - name: save_output
      type: SaveToDisk
      runtime_parameters:
        output_path: "./medqa-training/generate/generated_data.jsonl"
        format: "jsonl"
```

### Complex Reasoning with CoT Traces

```yaml
# medical_cot_pipeline.yaml

pipeline:
  name: medical_cot_generation

  steps:
    - name: load_documents
      type: LoadDataFromDisk
      runtime_parameters:
        data_path: "{SOURCE_PATH}"

    - name: chunk_text
      type: SentenceSplitter
      runtime_parameters:
        chunk_size: 800   # larger chunks for complex reasoning
        chunk_overlap: 100

    - name: generate_cot_qa
      type: TextGeneration
      llm:
        type: "{TEACHER_LLM_TYPE}"
        model: "{TEACHER_MODEL}"
        generation_kwargs:
          temperature: 0.7
          max_new_tokens: 1024
      runtime_parameters:
        system_prompt: |
          You are a medical educator creating training data for clinical reasoning.
          Given the passage below, generate a complex clinical question that requires
          multi-step diagnostic reasoning to answer.
          
          Format your response using these exact tags:
          Question: [complex clinical question]
          <think>
          [Step-by-step diagnostic reasoning: consider differentials, analyze evidence,
          apply clinical knowledge, arrive at conclusion. Minimum 150 words.]
          </think>
          <answer>[final answer letter if MCQ, or concise conclusion]</answer>
          
          The reasoning trace should model how an expert clinician thinks through the problem.

    - name: filter_cot_quality
      type: FilterByScore
      runtime_parameters:
        threshold: 0.65

    - name: save_output
      type: SaveToDisk
      runtime_parameters:
        output_path: "./medqa-training/generate/generated_cot_data.jsonl"
        format: "jsonl"
```

### DPO Preference Pairs (Chosen / Rejected)

```yaml
# medical_dpo_pipeline.yaml

pipeline:
  name: medical_dpo_generation

  steps:
    - name: load_documents
      type: LoadDataFromDisk
      runtime_parameters:
        data_path: "{SOURCE_PATH}"

    - name: chunk_text
      type: SentenceSplitter
      runtime_parameters:
        chunk_size: 512

    - name: generate_chosen
      type: TextGeneration
      llm:
        type: "{TEACHER_LLM_TYPE}"
        model: "{TEACHER_MODEL}"
        generation_kwargs:
          temperature: 0.3   # lower temp for chosen (more accurate)
          max_new_tokens: 512
      runtime_parameters:
        system_prompt: |
          You are a medical expert. Generate a clinically accurate, comprehensive answer
          to a question derived from this passage. Be precise, evidence-based, and safe.

    - name: generate_rejected
      type: TextGeneration
      llm:
        type: "{TEACHER_LLM_TYPE}"
        model: "{TEACHER_MODEL}"
        generation_kwargs:
          temperature: 1.2   # higher temp for rejected (more hallucination-prone)
          max_new_tokens: 512
      runtime_parameters:
        system_prompt: |
          Generate a medical answer that sounds plausible but contains one subtle clinical
          inaccuracy (wrong dosage, incorrect drug interaction, missed contraindication,
          or wrong diagnostic criterion). The error should be medically realistic but wrong.

    - name: save_dpo_pairs
      type: SaveToDisk
      runtime_parameters:
        output_path: "./medqa-training/generate/generated_dpo_pairs.jsonl"
        format: "jsonl"
```

---

## `generate.sh` Template

```bash
#!/bin/bash
# One-command synthetic data generator
# Teacher model auto-detected from session environment

set -euo pipefail

echo "Installing distilabel..."
pip install distilabel -q

echo "Auto-detecting teacher model..."
if [ -n "${CLAUDE_MODEL:-}" ]; then
    echo "  Teacher: $CLAUDE_MODEL (Claude Code session)"
elif [ -n "${OPENCODE_MODEL:-}" ]; then
    echo "  Teacher: $OPENCODE_MODEL (OpenCode session)"
else
    echo "  Teacher: claude-sonnet-4-6 (default fallback)"
fi

echo "Running distilabel pipeline..."
distilabel pipeline run --config medical_qa_pipeline.yaml

echo "Done! Output: ./medqa-training/generate/generated_data.jsonl"
```

---

## Scale Estimation

| Scale | Pairs | Approx time (claude-sonnet-4-6) | Cost estimate |
|-------|-------|--------------------------------|--------------|
| Small | 500–2K | 15–60 min | ~$0.50–2.00 |
| Medium | 5K–10K | 2–4 hours | ~$5–15 |
| Large | 10K+ | 4–8+ hours | ~$15–50+ |

Cost depends on passage length and whether quality scoring is enabled.

---

## Integration with Training Pipeline

After `generate.sh` completes, add generated data to Phase 3 dataset pool:

```python
from datasets import load_dataset, concatenate_datasets

# Load generated data
generated = load_dataset("json", data_files="./medqa-training/generate/generated_data.jsonl", split="train")

# Add to existing dataset
combined = concatenate_datasets([existing_dataset, generated])
```

The generated data is automatically included when the skill writes `train.py`.
