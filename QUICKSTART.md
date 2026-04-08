# Quick Start Guide

## What Changed

✅ **Tool Calling Format**: APIs are now defined as structured function calls (OpenAI-compatible format)
✅ **1-Second Delay**: Added `time.sleep(1.0)` after each task iteration  
✅ **Training Utilities**: Scripts for collecting trajectories, preparing data, and comparing models
✅ **Comprehensive Training Guide**: Full pipeline for training 0.8B models

## Usage

### 1. Test with Qwen (Baseline)

```bash
# Set up Ollama with Qwen
export OPENAI_API_KEY="ollama"
export INFERENCE_SERVER="http://localhost:11434/v1"
export MODEL_LOWER_NAME="qwen2.5:14b"

# Run evaluation (with tool calling)
python baseline_agent.py --model qwen2.5:14b --episodes 3 --output results/qwen.json

# Run with legacy JSON format
python baseline_agent.py --model qwen2.5:14b --episodes 3 --no-tools
```

### 2. Collect Training Data

```bash
# Collect 100 high-quality trajectories (grade >= 0.8)
python collect_trajectories.py \
  --model qwen2.5:14b \
  --episodes 100 \
  --min-grade 0.8 \
  --output data/qwen_trajectories.jsonl
```

### 3. Prepare Training Dataset

```bash
# Convert to OpenAI fine-tuning format (with tool calls)
python prepare_training_data.py \
  --input data/qwen_trajectories.jsonl \
  --output data/training_data.jsonl \
  --format openai

# Or convert to text format (for instruction tuning)
python prepare_training_data.py \
  --input data/qwen_trajectories.jsonl \
  --output data/training_data_text.jsonl \
  --format text
```

### 4. Train Your 0.8B Model

See `TRAINING_GUIDE.md` for detailed training instructions using:
- Unsloth (recommended for efficiency)
- TRL (for RLHF/DPO)
- OpenAI Fine-tuning API
- Custom training loops

### 5. Evaluate and Compare

```bash
# Evaluate your fine-tuned model
python baseline_agent.py \
  --model path/to/your/model \
  --episodes 10 \
  --output results/your_model.json

# Compare multiple models
python compare_models.py \
  results/qwen.json \
  results/your_sft_model.json \
  results/your_dpo_model.json \
  --names "Qwen 14B (Baseline)" "Your SFT 0.8B" "Your DPO 0.8B"
```

## Key Features

### Tool Calling Mode (Default)
- Uses OpenAI-compatible function calling format
- Better for training on tool use
- Structured parameter definitions
- Modern LLM paradigm

### Legacy JSON Mode (`--no-tools`)
- Falls back to prompt-based JSON responses
- For models without tool calling support
- Still works with same environment

### Automatic Delay
- 1-second delay after each task iteration
- Prevents overload on inference server
- Configurable in code if needed

## File Structure

```
APIOpenEnv/
├── baseline_agent.py              # Main agent (now with tool calling)
├── collect_trajectories.py        # Collect training data
├── prepare_training_data.py       # Convert to training format
├── compare_models.py               # Compare model performance
├── TRAINING_GUIDE.md               # Comprehensive training guide
├── QUICKSTART.md                   # This file
├── server/
│   ├── mock_apis.py                # API implementations
│   └── api_open_env_environment.py # Environment
└── data/                           # Training data (create this)
    ├── trajectories.jsonl
    └── training_data.jsonl
```

## Next Steps

1. **Test Current Setup**: Run baseline_agent.py with Qwen to verify everything works
2. **Collect Data**: Gather 200+ successful trajectories using collect_trajectories.py
3. **Prepare Dataset**: Convert to training format with prepare_training_data.py
4. **Train Model**: Follow TRAINING_GUIDE.md to train your 0.8B model
5. **Evaluate**: Compare your model against Qwen baseline
6. **Iterate**: Use DPO/RLHF to improve performance

## Tips

- Start with easy tasks only to verify pipeline works
- Collect more data for better performance (500+ trajectories recommended)
- Use LoRA for efficient training of small models
- Monitor overfitting - keep a validation set
- Experiment with different base models (Qwen-0.8B, Phi-2, etc.)

## Troubleshooting

**Tool calling not working?**
- Make sure your model supports function calling
- Use `--no-tools` flag for models without tool support

**Ollama connection issues?**
- Verify Ollama is running: `ollama list`
- Check INFERENCE_SERVER URL is correct
- Ensure MODEL_LOWER_NAME matches installed model

**Low grades in collected data?**
- Lower `--min-grade` threshold (try 0.6)
- Collect more episodes (increase --episodes)
- Focus on easy difficulty first

**Training taking too long?**
- Use LoRA instead of full fine-tuning
- Reduce batch size or sequence length
- Use Unsloth for faster training
- Consider cloud GPUs (Colab, Lambda, etc.)

## Questions?

See `TRAINING_GUIDE.md` for detailed explanations of:
- Training strategies (SFT → DPO → PPO)
- Data format specifications  
- Reward function design
- Evaluation metrics
- Best practices
