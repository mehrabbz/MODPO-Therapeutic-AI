# Reward Modeling

Train RoBERTa-based reward models on preference data for multi-objective therapeutic AI alignment.

## Overview

Reward models learn to predict human preferences by scoring responses. These models are used in:
- **MODPO training**: Provide margin guidance for multi-objective optimization
- **Evaluation**: Score model outputs on specific therapeutic criteria
- **Analysis**: Understand what makes responses preferred

## Usage

### Train Single Reward Model

```bash
python train_reward_model.py \
    --preferences ./preferences/empathy_pairs.json \
    --output_dir ./reward_models/empathy \
    --criterion empathy \
    --n_trials 20
```

### Train All Reward Models

```bash
bash train_all_reward_models.sh
```

### Skip Hyperparameter Optimization

For faster training with default/custom hyperparameters:

```bash
python train_reward_model.py \
    --preferences ./preferences/empathy_pairs.json \
    --output_dir ./reward_models/empathy \
    --criterion empathy \
    --skip_hpo \
    --learning_rate 2e-5 \
    --num_epochs 5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--preferences` | Required | Path to preference pairs JSON |
| `--output_dir` | Required | Output directory for model |
| `--criterion` | Required | Name of criterion (for logging) |
| `--model_name` | "roberta-large" | Base model |
| `--val_split` | 0.1 | Validation split ratio |
| `--n_trials` | 20 | Optuna HPO trials |
| `--skip_hpo` | False | Skip hyperparameter optimization |
| `--learning_rate` | 2e-5 | Learning rate (if skip_hpo) |
| `--weight_decay` | 0.03 | Weight decay (if skip_hpo) |
| `--dropout` | 0.1 | Dropout rate (if skip_hpo) |
| `--batch_size` | 8 | Batch size (if skip_hpo) |
| `--max_length` | 512 | Max sequence length |
| `--num_epochs` | 5 | Training epochs (if skip_hpo) |
| `--gpu_id` | 0 | GPU to use |

## Input Format

Expected preference pairs format (from `collect_preferences.py`):

```json
[
  {
    "question_id": "abc123",
    "question_text": "I've been feeling anxious...",
    "chosen": {
      "response_id": 3,
      "response_text": "I hear that you're experiencing..."
    },
    "rejected": {
      "response_id": 1,
      "response_text": "You should try to relax..."
    }
  }
]
```

## Output Structure

```
reward_models/
└── empathy/
    └── empathy_20240115_143022/
        ├── final_model/
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── tokenizer.json
        │   └── ...
        └── training_info.json
```

### training_info.json

```json
{
  "criterion": "empathy",
  "model_name": "roberta-large",
  "best_params": {
    "learning_rate": 1.5e-5,
    "weight_decay": 0.05,
    "dropout": 0.2,
    "batch_size": 8,
    "max_length": 512,
    "num_epochs": 5
  },
  "final_training_info": {
    "accuracy": 0.87
  },
  "data_size": 2379,
  "timestamp": "20240115_143022"
}
```

## Hyperparameter Search Space

When using HPO (`--n_trials > 0`), Optuna searches:

| Hyperparameter | Range |
|---------------|-------|
| learning_rate | 5e-7 to 2e-5 (log) |
| weight_decay | 0.01 to 0.15 (log) |
| dropout | 0.1 to 0.4 |
| batch_size | 8, 16 |
| max_length | 256, 512 |
| num_epochs | 3 to 7 |

## Training Details

- **Base Model**: RoBERTa-large (355M parameters)
- **Objective**: Binary classification (chosen > rejected)
- **Optimizer**: AdamW with cosine learning rate schedule
- **Evaluation Metric**: Accuracy on held-out validation set

## Using Trained Reward Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model_path = "./reward_models/empathy/empathy_20240115_143022/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Score a response
prompt = "I've been feeling anxious lately..."
response = "I hear that you're experiencing anxiety..."

inputs = tokenizer(
    prompt + " " + response,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    score = model(**inputs).logits.item()

print(f"Reward score: {score}")
```

## Hardware Requirements

- **GPU Memory**: ~16GB for RoBERTa-large with batch_size=8
- **Training Time**: ~30-60 min per criterion (with 20 HPO trials)
- **Storage**: ~1.5GB per trained model
