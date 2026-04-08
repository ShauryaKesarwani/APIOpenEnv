# 🚀 Quick Training Guide

## Prerequisites
```bash
# Install training dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft bitsandbytes datasets matplotlib
```

## 5-Step Training Process

### Step 1: Collect Expert Data (30-60 min)
```bash
export OPENAI_API_KEY="your-key"
# For Ollama:
# export INFERENCE_SERVER="http://localhost:11434/v1"
# export MODEL_LOWER_NAME="qwen2.5:14b"

python collect_trajectories.py \
  --model qwen2.5:14b \
  --episodes 100 \
  --min-grade 0.8 \
  --output data/qwen_trajectories.jsonl
```

**Output**: `data/qwen_trajectories.jsonl` (100+ successful task completions)

---

### Step 2: Prepare Dataset (1-2 min)
```bash
python prepare_training_data.py \
  --input data/qwen_trajectories.jsonl \
  --output data/training_data.jsonl \
  --format openai
```

**Output**: `data/training_data.jsonl` (formatted for fine-tuning)

---

### Step 3: Train Model (2-4 hours)
```bash
python train_model.py
```

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
python baseline_agent.py \
  --model qwen2.5:0.8b \
  --episodes 10 \
  --output results/base.json

# Test trained model
python baseline_agent.py \
  --model ./trained_model \
  --episodes 10 \
  --output results/trained.json
```

---

### Step 5: Compare Results (1 min)
```bash
python compare_models.py \
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

python inference.py
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
