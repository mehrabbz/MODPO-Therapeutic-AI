# Data

This folder contains scripts for data preparation.

## Required Dataset: EPITOME

Before running the scripts, you need to download the EPITOME dataset.

### Download Instructions

1. **Source:** The EPITOME dataset is from Sharma et al. (2020)
   
2. **Paper:** ["A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support"](https://aclanthology.org/2020.emnlp-main.425/) (EMNLP 2020)

3. **Download:** Get the dataset from the official repository:
   - https://github.com/behavioral-data/Empathy-Mental-Health

4. **Place the file:** After downloading, place `EPITOME.csv` in this folder or specify its path when running the script.

## Usage

```bash
# Basic usage
python prepare_dataset.py --input_path ./EPITOME.csv --output_dir ./processed

# Custom parameters
python prepare_dataset.py \
    --input_path ./EPITOME.csv \
    --output_dir ./processed \
    --test_size 600 \
    --random_state 42
```

## Output Files

The script generates:
- `questions_train.csv` - Training set (2,379 questions)
- `questions_test.csv` - Test set (600 questions)  
- `questions_all.csv` - Complete dataset (2,979 questions)

Each file contains:
| Column | Description |
|--------|-------------|
| `question_id` | Unique identifier (original `sp_id`) |
| `question_text` | The therapeutic question (seeker post) |
