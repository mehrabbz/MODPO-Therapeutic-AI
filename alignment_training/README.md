# Alignment Training

This module provides implementations of multiple alignment approaches for therapeutic AI:

1. **SFT** - Supervised Fine-Tuning on preferred responses
2. **DPO** - Direct Preference Optimization (single objective)
3. **MODPO** - Multi-Objective DPO with margin rewards
4. **Joint-Loss DPO** - Multi-objective DPO with combined loss
5. **DPO Soup** - Post-hoc parameter merging of DPO models

## Overview

| Method | Multi-Objective | Training Type | Key Feature |
|--------|----------------|---------------|-------------|
| SFT | No | Supervised | Learns from best responses only |
| DPO | No | Preference | Single criterion optimization |
| MODPO | Yes | Preference + Reward | Margin-based multi-objective |
| Joint-Loss DPO | Yes | Preference | Combined loss function |
| DPO Soup | Yes | Post-hoc | Parameter merging |

## Installation

```bash
pip install torch transformers trl peft bitsandbytes optuna safetensors
```

## Usage

### 1. Supervised Fine-Tuning (SFT)

Train on chosen responses for a single criterion:

```bash
python train_sft.py \
    --preferences ./preferences/empathy_pairs.json \
    --output_dir ./models/sft_empathy \
    --criterion empathy \
    --skip_hpo  # Optional: skip hyperparameter optimization
```

### 2. Direct Preference Optimization (DPO)

Standard DPO training on preference pairs:

```bash
python train_dpo.py \
    --preferences ./preferences/empathy_pairs.json \
    --output_dir ./models/dpo_empathy \
    --criterion empathy \
    --beta 0.1
```

### 3. Multi-Objective DPO (MODPO)

MODPO uses a primary preference dataset with margin rewards from auxiliary reward models:

```bash
python train_modpo.py \
    --preferences ./preferences/empathy_pairs.json \
    --reward_models safety:./reward_models/safety/final_model \
    --output_dir ./models/modpo_empathy \
    --primary_criterion empathy \
    --weights 0.5 0.5
```

**How MODPO works:**
- Primary objective (empathy) provides the main preference signal
- Margin objectives (safety) contribute via pre-computed reward differences
- The loss balances both through weighted combination

### 4. Joint-Loss DPO

Combines multiple preference objectives in a single training run:

```bash
python train_joint_loss_dpo.py \
    --preferences empathy:./preferences/empathy_pairs.json \
                  safety:./preferences/safety_pairs.json \
    --output_dir ./models/joint_loss_dpo \
    --weights 0.5 0.5
```

**How Joint-Loss DPO works:**
- Each objective has its own chosen/rejected pairs
- All objectives are combined in a single sigmoid-weighted loss
- Requires aligned preference data (same questions across objectives)

### 5. DPO Soup (Parameter Merging)

Merge separately trained DPO models post-hoc:

```bash
# First, train separate DPO models
python train_dpo.py --preferences ./preferences/empathy_pairs.json \
    --output_dir ./models/dpo_empathy --criterion empathy

python train_dpo.py --preferences ./preferences/safety_pairs.json \
    --output_dir ./models/dpo_safety --criterion safety

# Then merge them
python merge_dpo_models.py \
    --models empathy:./models/dpo_empathy/final_model \
             safety:./models/dpo_safety/final_model \
    --weights 0.5 0.5 \
    --output_dir ./models/dpo_soup
```

**Merge methods:**
- `linear`: Weighted average (default) - `θ_merged = Σ w_i * θ_i`
- `slerp`: Spherical interpolation (only for 2 models)

## Input Format

### Preference Pairs JSON

All methods expect preference pairs in this format:

```json
[
  {
    "question_id": "abc123",
    "question_text": "I've been feeling anxious lately...",
    "chosen": {
      "response_id": 3,
      "response_text": "I hear that you're experiencing anxiety..."
    },
    "rejected": {
      "response_id": 1,
      "response_text": "You should just try to relax..."
    }
  }
]
```

## Hyperparameter Optimization

All training scripts support automatic HPO via Optuna:

```bash
# Run with HPO (default)
python train_dpo.py --preferences ./prefs.json --output_dir ./out --criterion empathy

# Skip HPO and use default/specified params
python train_dpo.py --preferences ./prefs.json --output_dir ./out --criterion empathy \
    --skip_hpo --learning_rate 5e-6 --batch_size 4 --num_epochs 3

# Customize HPO trials
python train_dpo.py --preferences ./prefs.json --output_dir ./out --criterion empathy \
    --n_trials 50
```

**Hyperparameters searched:**
- Learning rate: 1e-7 to 1e-5 (log scale)
- Batch size: 2, 4, 8
- Epochs: 1-5
- Beta (DPO): 0.05-0.5
- LoRA rank: 32, 64, 128
- LoRA alpha: 64, 128, 256
- LoRA dropout: 0.01-0.1

## Output Structure

Each training run creates:

```
output_dir/
└── {method}_{criterion}_{timestamp}/
    ├── final_model/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── ...
    ├── training_info.json      # Hyperparameters and metrics
    └── checkpoint-*/           # Intermediate checkpoints
```

## Loading Trained Models

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./models/modpo_empathy_20240115/final_model"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./models/modpo_empathy_20240115/final_model"
)

# Generate
inputs = tokenizer("I've been feeling anxious...", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Method Comparison

From our paper's experiments:

| Method | Empathy Win Rate | Safety Win Rate |
|--------|-----------------|-----------------|
| Base (Mistral-7B) | 11.5% | 39.6% |
| SFT Empathy | 20.9% | 44.8% |
| DPO Empathy | 93.6% | 47.8% |
| DPO Soup | 52.5% | 58.4% |
| Joint-Loss DPO | 70.8% | 62.7% |
| **MODPO Empathy** | **77.6%** | **62.6%** |

**Key findings:**
- Single-objective DPO maximizes primary metric but neglects safety
- Multi-objective methods (MODPO, Joint-Loss) balance both dimensions
- MODPO achieves best overall balance through margin-based optimization

## Hardware Requirements

- **GPU Memory**: 24GB+ recommended (uses 4-bit quantization)
- **Training Time**: 
  - SFT: ~1-2 hours per criterion
  - DPO/MODPO: ~2-4 hours per criterion
  - With HPO (20 trials): ~24-48 hours

## References

- DPO: Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
- MODPO: Zhou et al., "Beyond One-Preference-Fits-All Alignment" (ACL 2024)
- DPO Soup: Jang et al., "Personalized Soups" (arXiv:2310.11564)
