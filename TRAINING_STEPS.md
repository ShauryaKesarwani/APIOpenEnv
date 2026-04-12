# 🚀 Quick Training Guide

## Prerequisites
```bash
# Install training dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft bitsandbytes datasets matplotlib
```

## 5-Step Training Process

### Step 1: Collect Expert Data (30-60 min)

PowerShell (Windows) + Ollama (Qwen3 0.8B):
```powershell
$env:INFERENCE_SERVER = "http://localhost:11434/v1"
$env:OPENAI_API_KEY   = "ollama"   # dummy key is fine for Ollama

# Quick sanity check (optional)
uv run --project . python baseline_agent.py --model qwen3:0.8b --episodes 1 --difficulty easy --no-tools

# Collect trajectories (use --no-tools if your Ollama build doesn't support tool_calls)
uv run --project . python collect_trajectories.py `
  --model qwen3:0.8b `
  --episodes 30 `
  --difficulty easy `
  --min-grade 0.6 `
  --no-tools `
  --output data/qwen3_0.8b_trajectories.jsonl
```

Notes:
- With only a 0.8B model as the "expert", you may need a lower `--min-grade` (0.5–0.7) to keep enough data.
- If you have access to a larger Ollama model, use that for collection (better demonstrations).

**Output**: `data/qwen_trajectories.jsonl` (100+ successful task completions)

---

### Step 2: Prepare Dataset (1-2 min)

This step is **optional** for `train_model.py` (that script trains directly from trajectories JSONL).
Use it if you want an OpenAI-style JSONL dataset for other trainers.

```powershell
uv run --project . python prepare_training_data.py `
  --input data/qwen3_0.8b_trajectories.jsonl `
  --output data/training_data_openai.jsonl `
  --format openai
```

**Output**: `data/training_data.jsonl` (formatted for fine-tuning)

---

### Step 3: Train Model (quick start)

`train_model.py` fine-tunes a **Hugging Face base model** with Unsloth (it does **not** fine-tune your Ollama model file).
In ~45 minutes you can do a short sanity run (few trajectories, 1 epoch):

```powershell
# Install training deps (can take a bit the first time)
python -m pip install -e ".[training]"

# Quick sanity training run
uv run --project . python train_model.py `
  --trajectories data/qwen3_0.8b_trajectories.jsonl `
  --epochs 1 `
  --min-grade 0.6 `
  --max-trajectories 30 `
  --output-dir trained_model_qwen3_sanity
```

If you specifically want a Qwen3 0.8B base (HF), pass `--base-model <hf_model_id>`.

**What it does**:
- Loads Qwen/Qwen2.5-0.8B-Instruct
- Adds LoRA adapters (rank=16)
- Trains for 3 epochs
- Saves to `./trained_model/`

**Hardware**: GPU with 8GB+ VRAM

---

### Step 4: Evaluate (10-20 min)
```bash
# Test base model
uv run --project . python baseline_agent.py \
  --model qwen2.5:0.8b \
  --episodes 10 \
  --output results/base.json

# Test trained model
uv run --project . python baseline_agent.py \
  --model ./trained_model \
  --episodes 10 \
  --output results/trained.json
```

---

### Step 5: Compare Results (1 min)
```bash
uv run --project . python compare_models.py \
  results/base.json \
  results/trained.json \
  --names "Qwen 0.8B Base" "Qwen 0.8B Trained"
```

**Output**: 
- Performance table in terminal
- `model_comparison.png` graph

---

## Expected Results

| Metric | Base 0.8B | Trained 0.8B | Target |
|--------|-----------|--------------|--------|
| Completion Rate | ~40% | **~65%** | 60%+ |
| Avg Grade | ~0.45 | **~0.65** | 0.60+ |
| Speed vs 14B | 10x faster | 10x faster | - |

**Success**: Achieve 60-70% of Qwen 14B performance at 17.5x smaller size!

---

## For Hackathon Submission

Use the standardized inference script:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-key"

uv run --project . python inference.py
```

This runs all 3 tasks with proper [START]/[STEP]/[END] logging.

---

## Troubleshooting

**"No trajectories found"**
- Lower `--min-grade` to 0.6
- Increase `--episodes` to 200

**"CUDA out of memory"**
- Use 4-bit quantization (already enabled)
- Reduce batch size in `train_model.py`

**"Model not improving"**
- Collect more data (300+ trajectories)
- Check data quality (should be grade >= 0.8)
- Train longer (5 epochs instead of 3)

---

## File Structure

```
APIOpenEnv/
├── inference.py              # Hackathon submission script
├── collect_trajectories.py   # Data collection
├── prepare_training_data.py  # Dataset preparation  
├── train_model.py            # LoRA fine-tuning
├── compare_models.py         # Evaluation
├── baseline_agent.py         # Run any model
└── data/                     # Training data (create this)
    ├── qwen_trajectories.jsonl
    └── training_data.jsonl
```

---

## Total Time

- Data Collection: 30-60 min
- Training: 2-4 hours
- Evaluation: 10-20 min

**Total**: ~3-5 hours from start to trained model
