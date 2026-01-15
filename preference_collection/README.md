# Preference Collection

Collect preferences by having LLM-based patient personas evaluate and rank therapeutic responses.

## Overview

This module uses patient personas to evaluate therapeutic responses according to configurable criteria. The personas "roleplay" as patients with specific characteristics and preferences, ranking responses based on how well they meet their therapeutic needs.

## Key Features

- **Configurable Criteria**: Evaluation criteria are defined in JSON config files
- **Modular Design**: Easy to add new criteria or use different evaluation frameworks
- **Persona-Based Evaluation**: Uses patient personas for authentic preference collection
- **Symmetrical Weighted Voting**: Aggregates rankings from multiple personas

## Usage

```bash
python collect_preferences.py \
    --config ./configs/therapeutic_criteria.json \
    --personas ./personas/train_personas.json \
    --responses ./responses/responses_train.csv \
    --output_dir ./preferences \
    --personas_per_question 50
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | Required | Path to criteria config JSON |
| `--personas` | Required | Path to personas JSON file |
| `--responses` | Required | Path to responses CSV file |
| `--output_dir` | "./preferences" | Output directory |
| `--personas_per_question` | 50 | Personas to sample per question |
| `--model` | "gpt-4o" | OpenAI model for evaluation |
| `--save_frequency` | 10 | Checkpoint save frequency |
| `--resume` | False | Resume from checkpoint |

## Config Files

### Available Configs

| Config | Description |
|--------|-------------|
| `therapeutic_criteria.json` | 6 criteria: 5 therapeutic + safety |
| `gricean_maxims.json` | 5 criteria: 4 Gricean maxims + safety |

### Config Structure

```json
{
  "config_name": "therapeutic_criteria",
  "description": "Description of the criteria set",
  "num_responses_to_rank": 5,
  "criteria": {
    "criterion_key": {
      "name": "Display Name",
      "short_name": "criterion_key",
      "description": "What this criterion measures",
      "better_when": "When responses rank higher",
      "worse_when": "When responses rank lower"
    }
  },
  "voting_weights": {
    "weights_5_responses": [10, 5, 0, -5, -10],
    "weights_10_responses": [45, 36, 27, 18, 9, -9, -18, -27, -36, -45]
  }
}
```

### Creating Custom Criteria

1. Copy an existing config file
2. Modify the criteria to match your evaluation framework
3. Run with `--config ./configs/your_criteria.json`

Example custom criterion:

```json
{
  "cultural_sensitivity": {
    "name": "Cultural Sensitivity",
    "short_name": "cultural_sensitivity",
    "description": "How well does the response respect cultural differences?",
    "better_when": "It acknowledges and respects diverse cultural backgrounds.",
    "worse_when": "It makes culturally insensitive assumptions."
  }
}
```

## Input Format

### Responses CSV

| Column | Description |
|--------|-------------|
| `question_id` | Unique question identifier |
| `question_text` | The therapeutic question |
| `response_id` | Response number (1 to N) |
| `response_text` | The generated response |

### Personas JSON

```json
{
  "persona_000": "You are roleplaying as a patient with...",
  "persona_001": "You are roleplaying as a patient with...",
  ...
}
```

## Output Format

For each criterion, two files are created:

### Preference Pairs (`{criterion}_pairs.json`)

Used for DPO training:

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

### Complete Rankings (`{criterion}_rankings.json`)

For analysis and reward model training:

```json
[
  {
    "question_id": "abc123",
    "question_text": "I've been feeling anxious...",
    "ranking": [
      {"rank": 1, "response_id": 3, "response_text": "..."},
      {"rank": 2, "response_id": 5, "response_text": "..."},
      ...
    ]
  }
]
```

## Voting Mechanism

Rankings from multiple personas are aggregated using **symmetrical weighted voting**:

For 5 responses:
- Rank 1 (best): +10 points
- Rank 2: +5 points
- Rank 3: 0 points
- Rank 4: -5 points
- Rank 5 (worst): -10 points

This ensures balanced positive/negative weighting and meaningful differentiation.

## Environment Setup

Requires OpenAI API access:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Therapeutic Criteria Reference

The default therapeutic criteria are based on clinical research:

1. **Empathy** - Warmth and compassion (Elliott et al., 2018)
2. **Active Listening** - Understanding and reflection (Reed et al., 2017)
3. **Self-Motivated Change** - Supporting readiness for change (Miller & Rose, 2009)
4. **Trust and Rapport** - Building therapeutic alliance (Horvath & Luborsky, 1991)
5. **Patient Autonomy** - Respecting self-determination (Ng et al., 2012)
6. **Safety** - Risk management and harm prevention (non-negotiable constraint)
