# Persona Generation

Generate patient personas from survey data for therapeutic AI evaluation.

## ⚠️ Privacy Notice

**Survey data files are NOT included in this repository** to protect participant privacy.

The scripts in this folder require survey response data that was collected under IRB approval. If you wish to replicate this work, you will need to:

1. Obtain IRB approval for your own survey
2. Collect survey responses following the format described below
3. Run the persona generation pipeline on your own data

## Pipeline Overview

The persona generation process involves three steps (combined into one script):

1. **Generation**: Convert raw survey responses → structured persona text
2. **Filtering**: Select high-quality, diverse personas
3. **Splitting**: Create balanced train/test sets

## Usage

**Single survey file:**
```bash
python generate_personas.py \
    --survey_files ./data/survey.csv \
    --output_dir ./personas \
    --num_personas 150
```

**Two survey files (with population stratification):**
```bash
python generate_personas.py \
    --survey_files ./data/survey_group_1.csv ./data/survey_group_2.csv \
    --labels "Group1" "Group2" \
    --output_dir ./personas \
    --num_personas 150 \
    --train_size 100 \
    --test_size 50
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--survey_files` | Required | Path(s) to survey CSV file(s). One or two files. |
| `--labels` | Auto | Label(s) for groups. Must match number of files. |
| `--output_dir` | "./personas" | Output directory |
| `--num_personas` | 150 | Total personas to select |
| `--train_size` | 100 | Training set size |
| `--test_size` | 50 | Test set size |
| `--min_quality` | 0.9 | Minimum quality score |
| `--random_state` | 42 | Random seed |

**Note:** When using two survey files, the train/test split will be stratified by population source. With a single file, only ethnicity/gender stratification is applied.

## Required Survey Columns

Your CSV files should contain these columns:

**Demographics:**
- `1. What is your age group?`
- `2. What is your gender?`
- `3. How would you describe your ethnicity/race? (Check all that apply)`

**Mental Health:**
- `4. Have you ever been diagnosed...` (diagnoses)
- `5. How would you describe the severity...`
- `6. Are you currently receiving treatment...`

**Technology Experience:**
- `6. Have you ever interacted with an automated system...` (LLM experience)

**AI Attitudes (7a-7e):**
- `7a. LLMs can help address mental health by offering accessible support.`
- `7b. LLMs can provide empathetic responses similar to human therapists.`
- `7c. I am concerned about LLMs making factual errors...`
- `7d. I believe LLMs should only be used as a complement...`
- `7e. To what extent do you trust LLMs...`

**Therapy Preferences (1-5 importance ratings):**
- `Empathy in responses (showing warmth and compassion)`
- `Active listening (understanding and reflecting the patient's feelings)`
- `Encouraging self-motivated change...`
- `Building trust and rapport...`
- `Respecting patient autonomy...`

**Use Cases:**
- `12. In your opinion, which areas of mental health...`
- `13. In which aspects of psychotherapy...`

## Output Files

| File | Description |
|------|-------------|
| `train_personas.json` | Training personas (persona_id → persona_text) |
| `test_personas.json` | Test personas (persona_id → persona_text) |
| `all_personas.json` | Complete data with metadata |
| `persona_report.txt` | Generation statistics |

## Persona Format

Each persona is a structured text that can be used as a system prompt:

```
You are roleplaying as a patient with the following characteristics:

DEMOGRAPHICS:
• Age: 25-34
• Gender: Female
• Ethnicity: Asian

MENTAL HEALTH BACKGROUND:
• Diagnoses: Depression, Anxiety
• Severity level: Moderate
• Current treatment: Talk therapy/Counseling

THERAPY PREFERENCES (importance ratings out of 5):
• Empathy and warmth: 5.0/5 importance
• Active listening: 4.0/5 importance
...
```

## Quality Filtering

Personas are scored based on:
- **Completeness** (40%): All required sections present
- **Consistency** (30%): Logically coherent responses
- **Specificity** (20%): Detailed, non-generic content
- **Validity** (10%): Proper formatting

## Demographic Balance

The train/test split ensures balanced representation across:
- Population groups (survey sources)
- Gender
- Ethnicity (with special attention to minority representation)
