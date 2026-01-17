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


<img width="1200" height="600" alt="phase1_safety_empathy_tradeoff" src="https://github.com/user-attachments/assets/c6985093-e755-47f0-ab07-7daa32ec0421" />


### Phase 2: Criteria Framework Comparison


<img width="2700" height="2100" alt="phase2_safety_overall_tradeoff" src="https://github.com/user-attachments/assets/a7937dc5-5be9-47c3-b01b-4d088c577006" />


## Hardware Requirements


- **Training**: 24GB+ GPU (A100/H100 recommended)
- **Inference**: 16GB+ GPU
- **Evaluation**: CPU only (uses OpenAI API)

Training times (per model, single A100):
- SFT: ~1-2 hours
- DPO/MODPO: ~2-4 hours
- With HPO (20 trials): ~24-48 hours


# Data Availability Section - Add to README.md

Insert this section after "## Results" and before "## Hardware Requirements":

---

## Data Availability

Due to the sensitive nature of mental health research data, the datasets used in this study (survey responses, preference rankings, patient personas) are **not publicly released**. 

### Available Upon Request

Anonymized data may be shared with qualified researchers who demonstrate proper credentials for human subjects research. Available datasets include:

| Dataset | Description | Size |
|---------|-------------|------|
| Patient Personas | Anonymized synthetic personas with demographics and therapeutic preferences | 150 |
| Preference Rankings | LLM-evaluated rankings across therapeutic criteria | ~119K |
| Therapeutic Q&A | Questions from EPITOME corpus with generated responses | 2,979 |
| **Model Responses** | **Generated responses from all trained models on test set** | **600 × 10 models** |
| Evaluation Results | Head-to-head model comparison results | 600 |

The model responses include outputs from: Base, GPT-4o, SFT Empathy, DPO Empathy, DPO Soup, Joint-Loss DPO, MODPO Empathy, MODPO Survey, MODPO Survey4, and MODPO Maxim. This enables replication of evaluation results without running inference.

### Request Process

To request data access, please:

1. **Review requirements**: See [DATA_REQUEST.md](DATA_REQUEST.md) for eligibility criteria
2. **Complete the form**: Download and fill out the [Data Request Form](docs/Data_Request_Form.docx)
3. **Submit with documentation**: Send to [INSERT EMAIL] with proof of CITI/equivalent training and IRB approval (if applicable)

**Requirements:**
- Institutional affiliation with university, research institution, or healthcare organization
- Current CITI or equivalent human subjects research training certification
- IRB approval or exemption documentation (if applicable)

Requests are typically reviewed within 2-4 weeks.

### Publicly Available Resources

The following are publicly available without data request:
- **Code**: All training, evaluation, and analysis code in this repository
- **EPITOME corpus**: Available from [Sharma et al. (2020)](https://github.com/behavioral-data/Empathy-Mental-Health)
- **Model architecture**: Based on Mistral-7B-Instruct-v0.2

---

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
