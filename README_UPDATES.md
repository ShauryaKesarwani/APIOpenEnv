# APIOpenEnv - Training Infrastructure Updates

## 🎯 Project Goal
Train a small 0.8B parameter model to predict correct API usage sequences, using Qwen as the upper-bound baseline for comparison.

## 🔄 What Changed

### ✅ Tool Calling Format (Primary Change)
**Before**: Free-form JSON responses in prompts
```json
{"api_name": "get_user", "args": {"user_id": "U101"}, "reasoning": "..."}
```

**After**: Structured OpenAI-compatible tool calling
```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_user",
            "description": "Retrieve user information...",
            "parameters": {...}
        }
    },
    ...
]
```

**Benefits**:
- 📊 Better for training small models (structured > unstructured)
- 🎓 Industry-standard format for function calling
- 🔍 Easier to evaluate correctness
- 🔄 Backward compatible (use `--no-tools` for legacy)

### ✅ 1-Second Delay
Added `time.sleep(1.0)` after each task iteration to prevent inference server overload.

### ✅ Training Infrastructure
Three new scripts for the complete training pipeline:
1. **collect_trajectories.py** - Gather training data from Qwen
2. **prepare_training_data.py** - Convert to fine-tuning format
3. **compare_models.py** - Evaluate and compare models

## 📊 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. BASELINE EVALUATION                       │
│  Run Qwen (14B) → Establish upper bound → Save trajectories     │
│  Command: python baseline_agent.py --model qwen2.5:14b          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. DATA COLLECTION                            │
│  Filter successful runs (grade >= 0.8) → 200+ trajectories      │
│  Command: python collect_trajectories.py --episodes 200         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. DATA PREPARATION                           │
│  Convert to tool calling format → Training dataset              │
│  Command: python prepare_training_data.py --format openai       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               4. SUPERVISED FINE-TUNING (SFT)                    │
│  Qwen2.5-0.8B + LoRA → Learn from expert trajectories           │
│  Tools: Unsloth, TRL, HuggingFace Transformers                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         5. DIRECT PREFERENCE OPTIMIZATION (DPO)                  │
│  Preference pairs (success vs failure) → Refine quality         │
│  Alternative: PPO with reward model                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               6. SELF-PLAY & AUTO-LEARNING                       │
│  Continuous improvement via environment feedback                │
│  Reward system: +10 completion, +2/correct call, -1/failure     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    7. EVALUATION & COMPARISON                    │
│  Compare 0.8B vs Qwen → Metrics: completion, grade, efficiency  │
│  Command: python compare_models.py results/*.json               │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Test with Qwen (Baseline)
```bash
export OPENAI_API_KEY="ollama"
export INFERENCE_SERVER="http://localhost:11434/v1"
export MODEL_LOWER_NAME="qwen2.5:14b"

python baseline_agent.py --model qwen2.5:14b --episodes 3 --output results/qwen.json
```

### 2. Collect Training Data
```bash
python collect_trajectories.py \
  --model qwen2.5:14b \
  --episodes 100 \
  --min-grade 0.8 \
  --output data/qwen_trajectories.jsonl
```

### 3. Prepare for Training
```bash
python prepare_training_data.py \
  --input data/qwen_trajectories.jsonl \
  --output data/training_data.jsonl \
  --format openai
```

### 4. Train (Example with Unsloth)
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.8B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", ...]
)

# Train!
trainer = SFTTrainer(model, tokenizer, train_dataset, ...)
trainer.train()
```

### 5. Evaluate Your Model
```bash
python baseline_agent.py \
  --model path/to/your/model \
  --episodes 10 \
  --output results/your_model.json

python compare_models.py \
  results/qwen.json results/your_model.json \
  --names "Qwen 14B" "Your 0.8B"
```

## 📁 New Files

| File | Purpose | Size |
|------|---------|------|
| `TRAINING_GUIDE.md` | Complete training tutorial | 12KB |
| `QUICKSTART.md` | Quick reference guide | 5KB |
| `CHANGES_SUMMARY.md` | Detailed changes explanation | 8KB |
| `collect_trajectories.py` | Data collection script | 7KB |
| `prepare_training_data.py` | Dataset preparation | 8KB |
| `compare_models.py` | Model comparison | 7KB |

## 🎓 Training Strategies Explained

### Supervised Fine-Tuning (SFT)
- Learn from successful Qwen trajectories
- Imitation learning: "Do what the expert does"
- Fast, stable, gets you 70-80% of the way there

### Direct Preference Optimization (DPO)
- Learn from comparisons: "This is better than that"
- Preferred: successful trajectories (grade > 0.8)
- Rejected: failed trajectories or inefficient sequences
- Simpler than PPO, works well for small models

### Proximal Policy Optimization (PPO)
- Learn from rewards via reinforcement learning
- Reward function based on environment feedback
- More complex but potentially higher performance
- Use after SFT + DPO if needed

### Self-Play Training
- Model improves by practicing on environment
- Collects its own training data
- Retrain periodically on recent successes
- Continuous improvement loop

## 📈 Expected Results

### Target Performance (0.8B model)
| Metric | Qwen 14B | Target 0.8B | Realistic 0.8B |
|--------|----------|-------------|----------------|
| Completion Rate | ~90% | 80%+ | 60-80% |
| Average Grade | ~0.85 | 0.70+ | 0.60-0.70 |
| Avg Steps | 4.5 | 5.0 | 4.5-5.5 |
| Parameters | 14B | 0.8B | 0.8B |
| **Efficiency** | 1x | **17.5x** | **17.5x** |

### Why This Matters
- **17.5x smaller** model
- **Much faster** inference (~10x)
- **Lower cost** to run
- **60-80%** of the performance

This is the holy grail: small models that punch above their weight class!

## 🔧 Configuration Options

### Tool Calling Mode (Default)
```bash
python baseline_agent.py --model qwen2.5:14b
```
Uses structured function calling format (recommended for training).

### Legacy JSON Mode
```bash
python baseline_agent.py --model qwen2.5:14b --no-tools
```
Uses free-form JSON responses (for models without tool support).

## 🐛 Troubleshooting

**"Tool calls not working"**
- Some models don't support tool calling yet
- Use `--no-tools` flag to fall back to JSON format
- Or train with text format: `--format text`

**"ModuleNotFoundError"**
```bash
pip install python-dotenv openai
```

**"Connection refused to Ollama"**
```bash
# Check Ollama is running
ollama list

# Verify server URL
export INFERENCE_SERVER="http://localhost:11434/v1"
```

**"Low grade in collected data"**
- Lower threshold: `--min-grade 0.6`
- Collect more: `--episodes 200`
- Start with easy only: `--difficulty easy`

## 📚 Further Reading

- **TRAINING_GUIDE.md**: Detailed training instructions
- **QUICKSTART.md**: Command reference
- **CHANGES_SUMMARY.md**: What changed and why

## 🎯 Success Criteria

Your 0.8B model is successful if it achieves:
- ✅ 60%+ completion rate on mixed difficulty
- ✅ 0.60+ average grade
- ✅ Efficient API sequences (< 6 steps average)
- ✅ Generalizes to unseen task variations

## 💡 Tips

1. **Start small**: Test pipeline with 10 episodes first
2. **Quality > Quantity**: Better 100 good trajectories than 1000 mediocre ones
3. **Monitor overfitting**: Keep validation set, check generalization
4. **Iterate**: SFT → evaluate → DPO → evaluate → repeat
5. **Curriculum learning**: Easy → Medium → Hard

## 🤝 Next Steps

1. [ ] Verify setup works with Qwen
2. [ ] Collect 200+ training trajectories
3. [ ] Prepare training dataset
4. [ ] Run SFT on 0.8B base model
5. [ ] Evaluate and compare to Qwen
6. [ ] Apply DPO for improvement
7. [ ] Document your results!

Good luck training your model! 🚀

---

**Questions?** See TRAINING_GUIDE.md for comprehensive explanations.
