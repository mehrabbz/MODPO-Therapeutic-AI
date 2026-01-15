# Evaluation

This module provides tools for evaluating trained therapeutic AI models through:
1. **Response Generation** - Generate responses from trained models on test questions
2. **Head-to-Head Evaluation** - Compare models using persona-based LLM evaluation with configurable criteria

## Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Trained Models    │────▶│ Generate Responses  │────▶│  Head-to-Head Eval  │
│  (from alignment/)  │     │  on test questions  │     │  using test personas│
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                     │                           │
                                     ▼                           ▼
                            {model}_responses.csv      summary_stats.json
                                                       {criterion}_results.json
```

## Directory Structure

```
evaluation/
├── generate_eval_responses.py     # Generate responses from models
├── generate_all_responses.sh      # Batch generation script
├── evaluate_head_to_head.py       # Head-to-head comparison
├── run_all_evaluations.sh         # Batch evaluation script
├── configs/
│   ├── therapeutic_criteria.json  # Default criteria (empathy, safety, overall)
│   ├── safety_only.json           # Safety-focused evaluation
│   └── full_therapeutic_survey.json  # All 7 therapeutic criteria
└── README.md
```

## Installation

```bash
pip install torch transformers peft pandas tqdm openai
```

For head-to-head evaluation, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Step 1: Generate Evaluation Responses

Generate responses from each trained model on the test set.

### Single Model

```bash
# LoRA adapter model
python generate_eval_responses.py \
    --model_path ./models/modpo_empathy/final_model \
    --model_name modpo_empathy \
    --test_questions ./data/questions_test.csv \
    --output_dir ./evaluation_responses

# Base model (no adapter)
python generate_eval_responses.py \
    --model_name base_model \
    --test_questions ./data/questions_test.csv \
    --output_dir ./evaluation_responses

# Merged model (DPO Soup)
python generate_eval_responses.py \
    --model_path ./models/dpo_empathy/final_model \
    --merge_with ./models/dpo_safety/final_model \
    --merge_weights 0.5 0.5 \
    --model_name dpo_soup \
    --test_questions ./data/questions_test.csv \
    --output_dir ./evaluation_responses
```

### All Models (Batch)

```bash
chmod +x generate_all_responses.sh
./generate_all_responses.sh 0  # GPU ID
```

### Output Format

Each model produces a CSV file:

| Column | Description |
|--------|-------------|
| `question_id` | Unique question identifier |
| `question_text` | The therapeutic question |
| `response_text` | Model's generated response |
| `model_name` | Name of the model |

## Step 2: Head-to-Head Evaluation

Compare two models using patient personas as evaluators with configurable criteria.

### How It Works

1. For each test question, both models' responses are shown to patient personas
2. Each persona ranks the responses based on therapeutic criteria (from config)
3. Winners are determined by majority voting across all personas
4. Results are aggregated across all questions

### Available Configs

| Config | Criteria | Use Case |
|--------|----------|----------|
| `therapeutic_criteria.json` | empathy, safety, overall_preference | Default evaluation |
| `safety_only.json` | safety | Safety-focused comparison |
| `full_therapeutic_survey.json` | All 7 criteria | Comprehensive evaluation |

### Single Comparison (Default Config)

```bash
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./evaluation_responses \
    --personas ./personas/test_personas.json \
    --output_dir ./evaluation_results
```

### Using Custom Config

```bash
# Full therapeutic criteria
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./evaluation_responses \
    --personas ./personas/test_personas.json \
    --output_dir ./evaluation_results \
    --config ./configs/full_therapeutic_survey.json

# Safety-only evaluation
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./evaluation_responses \
    --personas ./personas/test_personas.json \
    --output_dir ./evaluation_results \
    --config ./configs/safety_only.json
```

### Override Specific Criteria

```bash
# Evaluate only empathy and trust_rapport (must exist in config)
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./evaluation_responses \
    --personas ./personas/test_personas.json \
    --output_dir ./evaluation_results \
    --config ./configs/full_therapeutic_survey.json \
    --criteria empathy trust_rapport
```

### With Checkpointing (Recommended)

For long evaluations, use checkpointing to resume if interrupted:

```bash
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./evaluation_responses \
    --personas ./personas/test_personas.json \
    --output_dir ./evaluation_results \
    --resume
```

### All Pairs (Batch)

```bash
chmod +x run_all_evaluations.sh
./run_all_evaluations.sh --resume
```

## Evaluation Criteria

### Config File Format

Criteria are defined in JSON config files:

```json
{
  "config_name": "my_criteria",
  "description": "Description of this config",
  "criteria": {
    "criterion_key": {
      "name": "Display Name",
      "description": "What this criterion measures",
      "better_when": "When responses rank higher",
      "worse_when": "When responses rank lower"
    }
  },
  "default_criteria": ["criterion_key1", "criterion_key2"],
  "evaluation_instructions": "Custom instructions for evaluation prompt"
}
```

### Creating Custom Configs

1. Copy an existing config from `configs/`
2. Modify criteria to match your evaluation needs
3. Use with `--config ./configs/your_config.json`

### Available Criteria

| Criterion | Description |
|-----------|-------------|
| `empathy` | Warmth and compassion in responses |
| `safety` | Appropriate handling of safety concerns |
| `active_listening` | Reflecting and understanding emotional state |
| `self_motivated_change` | Supporting readiness for change |
| `trust_rapport` | Building therapeutic alliance |
| `patient_autonomy` | Respecting independence and decisions |
| `overall_preference` | Overall preference as a patient |

## Output Structure

```
evaluation_results/
└── {model1}_vs_{model2}/
    ├── summary_stats.json       # Win rates and statistics
    ├── empathy_results.json     # Per-question results for empathy
    ├── safety_results.json      # Per-question results for safety
    └── overall_preference_results.json
```

### Summary Stats Format

```json
{
  "model1": "modpo_empathy",
  "model2": "base_model",
  "total_questions": 600,
  "num_personas": 50,
  "empathy": {
    "modpo_empathy_wins": 465,
    "base_model_wins": 120,
    "ties": 15,
    "modpo_empathy_winrate": 77.5,
    "base_model_winrate": 20.0
  },
  "safety": {
    "modpo_empathy_wins": 375,
    "base_model_wins": 180,
    "ties": 45,
    "modpo_empathy_winrate": 62.5,
    "base_model_winrate": 30.0
  }
}
```

## Statistical Analysis

After running evaluations, statistical significance can be computed using McNemar's test:

```python
from scipy.stats import mcnemar
import json

# Load results
with open("evaluation_results/modpo_empathy_vs_base_model/empathy_results.json") as f:
    results = json.load(f)

# Build contingency table
# Count questions where each model won
model1_wins = sum(1 for r in results if r["winner"] == "modpo_empathy")
model2_wins = sum(1 for r in results if r["winner"] == "base_model")

# McNemar's test (simplified - for full analysis, track per-question outcomes)
result = mcnemar([[0, model1_wins], [model2_wins, 0]], exact=False)
print(f"McNemar's chi-squared: {result.statistic:.2f}, p-value: {result.pvalue:.4f}")
```

## Toxicity Evaluation

Toxicity can be evaluated using external tools:

- **Perspective API**: https://perspectiveapi.com/
- **ModelCitizens**: https://github.com/anthropics/model-citizens

Example with Perspective API:

```python
from googleapiclient import discovery

client = discovery.build("commentanalyzer", "v1alpha1", 
                         developerKey="YOUR_API_KEY")

def get_toxicity(text):
    response = client.comments().analyze(
        body={"comment": {"text": text}, 
              "requestedAttributes": {"TOXICITY": {}}}
    ).execute()
    return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
```

## Hardware Requirements

- **Response Generation**: 16GB+ GPU (uses half precision)
- **Head-to-Head Evaluation**: No GPU needed (uses OpenAI API)
- **API Costs**: ~$0.01-0.05 per question (depends on model and personas)

## Tips

1. **Start small**: Test with 10 questions and 5 personas first
2. **Use checkpointing**: Always use `--resume` for long evaluations
3. **Monitor API costs**: Track OpenAI usage during evaluation
4. **Parallelize carefully**: Too many parallel jobs may hit rate limits
