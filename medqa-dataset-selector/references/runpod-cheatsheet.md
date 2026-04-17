# RunPod Cheatsheet

Quick reference for running Unsloth medical LLM training on RunPod.

---

## GPU Tier Table (April 2026 pricing — verify at runpod.io)

| GPU | VRAM | Spot price | On-demand | Best for |
|-----|------|-----------|-----------|---------|
| RTX 4090 | 24GB | ~$0.44/hr | ~$0.88/hr | **Default recommendation** — great price/performance for 7–9B models, GRPO |
| A6000 | 48GB | ~$0.23/hr | ~$0.49/hr | **Best value** for medium models (13–27B), DPO runs |
| A100 80GB | 80GB | ~$1.89/hr | ~$2.72/hr | Large models, full GRPO curriculum A+B+C |
| H100 80GB | 80GB | ~$2.99/hr | ~$4.18/hr | Enterprise, fastest training |
| RTX 3090 | 24GB | ~$0.22/hr | ~$0.44/hr | Budget option, slower than 4090 |

**Recommendation for `/medqa` workflow**:
- 7–9B models (Qwen3.5-9B, Gemma-4-E4B): RTX 4090 spot
- 27B models (Qwen3.5-27B): A6000 spot
- GRPO full curriculum: A100 80GB

---

## Typical Job Cost (RTX 4090 Spot)

| Task | Duration | Estimated Cost |
|------|----------|---------------|
| SFT, 10K examples | ~1.5 hr | ~$0.66 |
| SFT, 50K examples | ~6 hr | ~$2.64 |
| GRPO Phase A (2K Q) | ~3 hr | ~$1.32 |
| GRPO Full A+B+C | ~10 hr | ~$4.40 |
| DPO, 10K pairs | ~2 hr | ~$0.88 |

---

## Pod Launch Flow (5 Steps)

1. **Log in** → runpod.io → Pods → Deploy
2. **Choose GPU** → filter by VRAM tier → select spot for cost savings
3. **Template** → select `RunPod Pytorch 2.x` (has CUDA pre-installed)
4. **Volume** → attach a Network Volume (min 50GB for model + dataset) at `/workspace`
5. **Deploy** → SSH in with: `ssh root@{pod-ip} -p {port}`

---

## Storage Rules (READ CAREFULLY)

- **Network volume**: Persistent storage at `/workspace` — survives pod termination. Cost: $0.07/GB/month.
- **Container disk**: Temporary — cleared when pod stops. **Never save models here.**
- **CANNOT shrink** a network volume after creation. Size up generously from the start.
- Minimum recommended: 50GB network volume for one 7B model + datasets + checkpoints.

```bash
# Always verify you're saving to network volume
ls /workspace  # should show your persisted files
```

---

## Unsloth Install on RunPod

```bash
# Use uv (much faster than pip on RunPod)
pip install uv -q
uv pip install unsloth

# If CUDA version mismatch, pin torch version first:
# Check current CUDA:
python -c "import torch; print(torch.version.cuda)"

# Example for CUDA 12.1:
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install unsloth

# Qwen3.5 additional requirement:
pip install transformers>=5.0.0
```

---

## Spot Training Best Practices

Spot pods can be terminated with 30-second warning. Protect your training run:

```python
# In your train.py — these are critical for spot instances
trainer = SFTTrainer(
    args=SFTConfig(
        save_steps=100,                    # checkpoint every 100 steps
        resume_from_checkpoint=True,        # auto-resume if checkpoint exists
        output_dir="/workspace/outputs",   # save to network volume!
    )
)
```

```bash
# Before starting training, confirm output dir is on network volume
ls /workspace  # should work

# Start training in tmux (survives SSH disconnection)
tmux new -s train
python train.py
# Ctrl+B, D to detach; tmux attach -t train to reattach
```

---

## 10-Step Quick Workflow

```bash
# 1. SSH into pod
ssh root@{pod-ip} -p {port}

# 2. Verify network volume is mounted
ls /workspace

# 3. Install dependencies
cd /workspace
pip install uv -q && uv pip install unsloth
pip install transformers>=5.0.0  # for Qwen3.5
pip install datasets trl

# 4. Download training data
python -c "
from datasets import load_dataset
ds = load_dataset('GBaker/MedQA-USMLE-4-options')
ds.save_to_disk('/workspace/medqa_usmle')
"

# 5. Upload your generated train.py
scp -P {port} ./medqa-training/train.py root@{pod-ip}:/workspace/

# 6. Start training in tmux
tmux new -s train
cd /workspace && python train.py

# 7. Monitor GPU usage (in another terminal)
watch -n 5 nvidia-smi

# 8. After training, export GGUF
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('./final_model')
model.save_pretrained_gguf('./final_model_gguf', tokenizer, quantization_method='q4_k_m')
"

# 9. Download results to local machine
rsync -avz -e "ssh -p {port}" root@{pod-ip}:/workspace/final_model_gguf ./local_model/

# 10. STOP the pod (not just terminate — stop = keep network volume)
# runpod.io UI → Pod → Stop
```

---

## Critical Gotchas

| Mistake | Consequence | Prevention |
|---------|------------|-----------|
| **Terminating pod** | ALL data lost (network volume detaches but data survives; container disk gone) | Use Stop, not Terminate. Keep checkpoints on /workspace |
| **Saving to container disk** | Data gone when pod stops | Always use `/workspace/` prefix |
| **No tmux** | Training dies when SSH disconnects | Always use `tmux new -s train` |
| **CUDA version mismatch** | Import error / cryptic failures | Check `torch.version.cuda` before installing Unsloth |
| **No inactivity timeout** | Runaway charges if you forget to stop | Set inactivity timeout 1–2 hours in pod settings |
| **Network volume too small** | Out of disk space mid-training | Allocate 50GB minimum; cannot shrink after creation |
| **Spot terminated mid-GRPO** | Wasted compute | Enable save_steps=100 + resume_from_checkpoint |

---

## Cost Monitoring

```bash
# Check approximate cost so far (RunPod CLI)
runpodctl get pod {pod-id}

# Manual estimate: time * hourly rate
# RTX 4090 spot: $0.44/hr
# 8 hours training: ~$3.52
```

Set a budget alert in RunPod account settings → Billing → Alerts.
