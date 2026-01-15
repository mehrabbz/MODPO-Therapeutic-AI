# Response Generation

Generate therapeutic responses using Mistral-7B-Instruct-v0.2.

## Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `generate_responses.py` | Single-GPU version | Most users |
| `generate_responses_parallel.py` | Multi-GPU with checkpointing | HPC clusters |

## Quick Start (Single GPU)

```bash
python generate_responses.py \
    --input_path ./processed/questions_train.csv \
    --output_path ./processed/responses_train.csv \
    --num_responses 5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_responses` | 5 | Responses per question |
| `--temperature` | 0.8 | Sampling temperature |
| `--max_new_tokens` | 512 | Max tokens to generate |
| `--batch_size` | 8 | Generation batch size |
| `--checkpoint_every` | 100 | Checkpoint frequency |
| `--resume` | False | Resume from checkpoint |

## Multi-GPU Usage

For multi-GPU setups (e.g., HPC clusters):

```bash
# Edit configuration in the script first
bash run_parallel_generation.sh
```

Or run manually on specific GPUs:

```bash
# GPU 0
python generate_responses_parallel.py chunk_0.csv --gpu_id 0 --output_dir ./responses &

# GPU 1
python generate_responses_parallel.py chunk_1.csv --gpu_id 1 --output_dir ./responses &
```

## Output Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `question_id` | Original question identifier |
| `question_text` | The therapeutic question |
| `response_id` | Response number (1 to num_responses) |
| `response_text` | Generated therapeutic response |

## Hardware Requirements

- **Single GPU:** ~16GB VRAM (e.g., RTX 4090, A100)
- **Multi-GPU:** Scales linearly with number of GPUs
- **CPU-only:** Possible but very slow (not recommended)

## Resume Functionality

If generation is interrupted, use `--resume` to continue:

```bash
python generate_responses.py \
    --input_path ./processed/questions_train.csv \
    --output_path ./processed/responses_train.csv \
    --resume
```
