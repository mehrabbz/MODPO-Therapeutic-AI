# MODPO: Multi-Objective Alignment for Personalized Psychotherapy

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **"Multi-Objective Alignment of Language Models for Personalized Psychotherapy"**.

## Overview

This repository provides a framework for training therapeutic AI models that balance multiple competing objectives (empathy, safety, active listening, etc.) using Multi-Objective Direct Preference Optimization (MODPO).


<img width="3208" height="2408" alt="phase1_methodology_pipeline" src="https://github.com/user-attachments/assets/f6e44487-0aa1-4255-8938-920164c28061" />



<img width="3208" height="2608" alt="phase2_methodology_pipeline" src="https://github.com/user-attachments/assets/df1b21e3-a5ee-4427-8049-de67b6aba3c9" />

**Key Findings:**
- Single-objective optimization maximizes one criterion but neglects others (e.g., 93.6% empathy but only 47.8% safety)
- MODPO achieves balanced performance across dimensions (77.6% empathy, 62.6% safety)
- Therapeutic-specific criteria outperform general communication principles by 17.2%

![Safety-Empathy Trade-off](docs/figures/safety_empathy_tradeoff.png)

## Installation

```bash
git clone https://github.com/yourusername/MODPO-Therapeutic-AI.git
cd MODPO-Therapeutic-AI
pip install -r requirements.txt
```

## Repository Structure

```
MODPO-Therapeutic-AI/
├── data/                      # Dataset preparation
│   ├── prepare_dataset.py     # Process EPITOME corpus
│   └── README.md
├── response_generation/       # Generate therapeutic responses
│   ├── generate_responses.py
│   └── README.md
├── persona_generation/        # Create patient personas from surveys
│   ├── generate_personas.py
│   └── README.md
├── preference_collection/     # Collect preferences via LLM evaluation
│   ├── collect_preferences.py
│   ├── configs/
│   │   ├── therapeutic_criteria.json
│   │   └── gricean_maxims.json
│   └── README.md
├── reward_modeling/           # Train reward models per criterion
│   ├── train_reward_model.py
│   └── README.md
├── alignment_training/        # Train aligned models
│   ├── train_sft.py          # Supervised Fine-Tuning
│   ├── train_dpo.py          # Direct Preference Optimization
│   ├── train_modpo.py        # Multi-Objective DPO
│   ├── train_joint_loss_dpo.py
│   ├── merge_dpo_models.py   # DPO Soup (parameter merging)
│   ├── trainers/
│   │   ├── modpo_trainer.py
│   │   └── joint_loss_trainer.py
│   └── README.md
├── evaluation/                # Evaluate trained models
│   ├── generate_eval_responses.py
│   ├── evaluate_head_to_head.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Dataset

```bash
# Download EPITOME dataset from https://github.com/behavioral-data/Empathy-Mental-Health
cd data
python prepare_dataset.py --input_path ./EPITOME.csv --output_dir ./processed
```

### 2. Generate Responses

```bash
cd response_generation
python generate_responses.py \
    --questions ../data/processed/questions_train.csv \
    --output_dir ./responses \
    --num_responses 5
```

### 3. Collect Preferences

```bash
cd preference_collection
python collect_preferences.py \
    --responses ../response_generation/responses/responses.csv \
    --personas ../personas/train_personas.json \
    --config ./configs/therapeutic_criteria.json \
    --output_dir ./preferences
```

### 4. Train Reward Models

```bash
cd reward_modeling
python train_reward_model.py \
    --preferences ../preference_collection/preferences/empathy_rankings.json \
    --output_dir ./reward_models/empathy \
    --criterion empathy
```

### 5. Train MODPO Model

```bash
cd alignment_training
python train_modpo.py \
    --preferences ../preference_collection/preferences/empathy_pairs.json \
    --reward_models safety:../reward_modeling/reward_models/safety/final_model \
    --output_dir ./models/modpo_empathy \
    --primary_criterion empathy \
    --weights 0.5 0.5
```

### 6. Evaluate

```bash
cd evaluation

# Generate responses from trained model
python generate_eval_responses.py \
    --model_path ../alignment_training/models/modpo_empathy/final_model \
    --model_name modpo_empathy \
    --test_questions ../data/processed/questions_test.csv \
    --output_dir ./responses

# Head-to-head comparison
python evaluate_head_to_head.py \
    --model1 modpo_empathy \
    --model2 base_model \
    --responses_dir ./responses \
    --personas ../personas/test_personas.json \
    --output_dir ./results
```

## Methods

### Alignment Approaches

| Method | Description | Multi-Objective |
|--------|-------------|-----------------|
| **SFT** | Supervised fine-tuning on preferred responses | No |
| **DPO** | Direct Preference Optimization | No |
| **MODPO** | Multi-Objective DPO with margin rewards | Yes |
| **Joint-Loss DPO** | Combined loss across multiple objectives | Yes |
| **DPO Soup** | Post-hoc parameter merging | Yes |

### Therapeutic Criteria

Based on clinical research, we optimize for:
- **Empathy** - Warmth and compassion (Elliott et al., 2018)
- **Safety** - Risk management and harm prevention
- **Active Listening** - Reflecting emotional states (Reed et al., 2017)
- **Self-Motivated Change** - Supporting readiness for change (Miller & Rose, 2009)
- **Trust and Rapport** - Building therapeutic alliance (Horvath & Luborsky, 1991)
- **Patient Autonomy** - Respecting self-determination (Ng et al., 2012)

## Results

### Phase 1: Training Methodology Comparison

| Model | Empathy Win Rate | Safety Win Rate |
|-------|-----------------|-----------------|
| Base (Mistral-7B) | 11.5% | 39.6% |
| SFT Empathy | 20.9% | 44.8% |
| DPO Empathy | 93.6% | 47.8% |
| DPO Soup | 52.5% | 58.4% |
| Joint-Loss DPO | 70.8% | 62.7% |
| **MODPO Empathy** | **77.6%** | **62.6%** |

### Phase 2: Criteria Framework Comparison

| Model | Overall Preference | Safety |
|-------|-------------------|--------|
| Base | 1.1% | 46.8% |
| MODPO Maxim | 56.9% | 51.6% |
| MODPO Survey4 | 67.1% | 49.1% |
| **MODPO Survey** | **74.1%** | **52.3%** |

## Hardware Requirements

- **Training**: 24GB+ GPU (A100/H100 recommended)
- **Inference**: 16GB+ GPU
- **Evaluation**: CPU only (uses OpenAI API)

Training times (per model, single A100):
- SFT: ~1-2 hours
- DPO/MODPO: ~2-4 hours
- With HPO (20 trials): ~24-48 hours

## Citation

```bibtex
@article{beikzadeh2025modpo,
  title={Multi-Objective Alignment of Language Models for Personalized Psychotherapy},
  author={Beikzadeh, Mehrab and Malgaroli, Matteo and Gabriel, Saadia},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- EPITOME dataset from Sharma et al. (2020)
- MODPO methodology from Zhou et al. (2024)
- DPO Soup from Jang et al. (2023)

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
