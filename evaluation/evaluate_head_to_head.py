"""
Head-to-Head Model Evaluation

Compares two models using persona-based LLM evaluation.
Each question is evaluated by multiple patient personas who rank responses
based on therapeutic criteria defined in a config file.

Usage:
    # Using default criteria config
    python evaluate_head_to_head.py \
        --model1 modpo_empathy \
        --model2 base_model \
        --responses_dir ./evaluation_responses \
        --personas ./personas/test_personas.json \
        --output_dir ./evaluation_results

    # Using custom criteria config
    python evaluate_head_to_head.py \
        --model1 modpo_empathy \
        --model2 base_model \
        --responses_dir ./evaluation_responses \
        --personas ./personas/test_personas.json \
        --output_dir ./evaluation_results \
        --config ./configs/full_therapeutic_survey.json

    # With checkpointing (resume if interrupted)
    python evaluate_head_to_head.py \
        --model1 modpo_empathy \
        --model2 base_model \
        --responses_dir ./evaluation_responses \
        --personas ./personas/test_personas.json \
        --output_dir ./evaluation_results \
        --resume
"""

import argparse
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "therapeutic_criteria.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head model evaluation")
    
    # Model comparison
    parser.add_argument("--model1", type=str, required=True, help="First model name")
    parser.add_argument("--model2", type=str, required=True, help="Second model name")
    
    # Input files
    parser.add_argument("--responses_dir", type=str, required=True,
                        help="Directory containing model response CSVs")
    parser.add_argument("--personas", type=str, required=True,
                        help="Path to test personas JSON")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to evaluation criteria config JSON (default: configs/therapeutic_criteria.json)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    # Evaluation settings
    parser.add_argument("--criteria", type=str, nargs='+', default=None,
                        help="Override: specific criteria to evaluate (default: use config's default_criteria)")
    parser.add_argument("--num_personas", type=int, default=50,
                        help="Number of personas to use for evaluation")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4o",
                        help="OpenAI model for evaluation")
    
    # Checkpointing
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                        help="Save checkpoint every N questions")
    
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load evaluation criteria config."""
    if config_path is None:
        config_path = DEFAULT_CONFIG
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded config: {config.get('config_name', 'unknown')}")
    logger.info(f"Available criteria: {list(config['criteria'].keys())}")
    
    return config


def load_responses(responses_dir: str, model_name: str) -> pd.DataFrame:
    """Load response CSV for a model."""
    response_file = Path(responses_dir) / f"{model_name}_responses.csv"
    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    return pd.read_csv(response_file)


def load_personas(personas_path: str, num_personas: int) -> dict:
    """Load patient personas."""
    with open(personas_path, 'r') as f:
        all_personas = json.load(f)
    
    # Take first N personas
    persona_ids = list(all_personas.keys())[:num_personas]
    return {pid: all_personas[pid] for pid in persona_ids}


def build_criteria_prompt(criteria_keys: list[str], config: dict) -> str:
    """Build evaluation criteria section of prompt from config."""
    criteria_text = ""
    for key in criteria_keys:
        if key not in config["criteria"]:
            logger.warning(f"Criterion '{key}' not found in config, skipping")
            continue
        
        c = config["criteria"][key]
        criteria_text += f"""
**{c['name']}** – {c['description']}
   - Better ranks when: {c['better_when']}
   - Worse ranks when: {c['worse_when']}
"""
    return criteria_text


def create_evaluation_prompt(
    persona: str,
    question: str,
    response_a: str,
    response_b: str,
    criteria_keys: list[str],
    config: dict
) -> str:
    """Create evaluation prompt for a persona using config."""
    
    criteria_text = build_criteria_prompt(criteria_keys, config)
    
    # Build expected JSON structure
    json_structure = "{\n"
    for key in criteria_keys:
        if key in config["criteria"]:
            json_structure += f'  "{key}_ranking": ["A", "B"] or ["B", "A"],\n'
    json_structure += '  "explanation": "Brief explanation reflecting your persona characteristics"\n}'
    
    # Get evaluation instructions from config or use default
    eval_instructions = config.get(
        "evaluation_instructions",
        "Based on your characteristics and preferences described above, please evaluate these responses."
    )
    
    prompt = f"""{persona}

EVALUATION TASK:
Imagine you are in a therapy session and you asked the following question:

\"{question}\"

You received these different responses from two therapists:

Response A:
{response_a}

Response B:
{response_b}

{eval_instructions}

Please evaluate on the following criteria:
{criteria_text}

Return your answer in JSON format using the following structure:

{json_structure}
"""
    return prompt


def get_evaluation(prompt: str, model: str = "gpt-4o", max_retries: int = 3) -> dict:
    """Call OpenAI API to get evaluation."""
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
        except openai.RateLimitError:
            wait_time = 2 ** attempt
            logger.warning(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            if "quota" in str(e).lower():
                raise Exception("QUOTA_EXCEEDED")
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    
    raise Exception(f"Failed to get evaluation after {max_retries} attempts")


def evaluate_question(
    personas: dict,
    question: str,
    response_1: str,
    response_2: str,
    model1: str,
    model2: str,
    criteria_keys: list[str],
    config: dict,
    evaluator_model: str
) -> dict:
    """Evaluate a single question with all personas."""
    
    evaluations = {}
    
    # Randomize response order (A/B assignment)
    if random.random() < 0.5:
        response_a, response_b = response_1, response_2
        order = [model1, model2]
    else:
        response_a, response_b = response_2, response_1
        order = [model2, model1]
    
    for persona_id, persona_text in tqdm(personas.items(), desc="Personas", leave=False):
        prompt = create_evaluation_prompt(
            persona_text, question, response_a, response_b, 
            criteria_keys, config
        )
        
        try:
            evaluation = get_evaluation(prompt, model=evaluator_model)
            evaluations[persona_id] = {
                "evaluation": evaluation,
                "response_order": order
            }
        except Exception as e:
            if "QUOTA_EXCEEDED" in str(e):
                raise
            logger.warning(f"Failed for persona {persona_id}: {e}")
            continue
        
        time.sleep(0.3)  # Rate limiting
    
    return evaluations


def calculate_winner(
    evaluations: dict,
    criterion: str,
    model1: str,
    model2: str
) -> dict:
    """Calculate winner for a criterion based on majority voting."""
    
    model1_votes = 0
    model2_votes = 0
    
    ranking_key = f"{criterion}_ranking"
    
    for persona_id, eval_data in evaluations.items():
        evaluation = eval_data["evaluation"]
        order = eval_data["response_order"]
        
        if ranking_key not in evaluation:
            continue
        
        preferred = evaluation[ranking_key][0]  # "A" or "B"
        
        # Map letter to model based on order
        if preferred == "A":
            winner_model = order[0]
        else:
            winner_model = order[1]
        
        if winner_model == model1:
            model1_votes += 1
        else:
            model2_votes += 1
    
    total = model1_votes + model2_votes
    
    if total == 0:
        return {
            "winner": "tie",
            "model1_votes": 0,
            "model2_votes": 0,
            "model1_percent": 0,
            "model2_percent": 0
        }
    
    if model1_votes > model2_votes:
        winner = model1
    elif model2_votes > model1_votes:
        winner = model2
    else:
        winner = "tie"
    
    return {
        "winner": winner,
        "model1_votes": model1_votes,
        "model2_votes": model2_votes,
        "model1_percent": (model1_votes / total) * 100,
        "model2_percent": (model2_votes / total) * 100
    }


def load_checkpoint(output_dir: str, model1: str, model2: str) -> dict:
    """Load checkpoint if exists."""
    checkpoint_file = Path(output_dir) / f"{model1}_vs_{model2}" / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(output_dir: str, model1: str, model2: str, data: dict):
    """Save checkpoint."""
    checkpoint_dir = Path(output_dir) / f"{model1}_vs_{model2}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_dir / "checkpoint.json", 'w') as f:
        json.dump(data, f, indent=2)


def save_results(output_dir: str, model1: str, model2: str, results: dict, 
                 summary: dict, config: dict):
    """Save final results."""
    result_dir = Path(output_dir) / f"{model1}_vs_{model2}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-criterion results
    for criterion, data in results.items():
        with open(result_dir / f"{criterion}_results.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    # Save summary
    with open(result_dir / "summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save config used for this evaluation
    with open(result_dir / "evaluation_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Remove checkpoint
    checkpoint_file = result_dir / "checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def main():
    args = parse_args()
    
    logger.info(f"Evaluating: {args.model1} vs {args.model2}")
    
    # Load config
    config = load_config(args.config)
    
    # Determine which criteria to use
    if args.criteria:
        # Override from command line
        criteria_keys = args.criteria
    else:
        # Use default from config
        criteria_keys = config.get("default_criteria", list(config["criteria"].keys()))
    
    # Validate criteria exist in config
    valid_criteria = [k for k in criteria_keys if k in config["criteria"]]
    if len(valid_criteria) != len(criteria_keys):
        invalid = set(criteria_keys) - set(valid_criteria)
        logger.warning(f"Invalid criteria (not in config): {invalid}")
    criteria_keys = valid_criteria
    
    logger.info(f"Evaluating on criteria: {criteria_keys}")
    
    # Load data
    logger.info("Loading responses...")
    responses_1 = load_responses(args.responses_dir, args.model1)
    responses_2 = load_responses(args.responses_dir, args.model2)
    
    # Merge on question_id
    merged = responses_1.merge(
        responses_2, 
        on="question_id", 
        suffixes=(f"_{args.model1}", f"_{args.model2}")
    )
    logger.info(f"Loaded {len(merged)} questions")
    
    # Load personas
    logger.info("Loading personas...")
    personas = load_personas(args.personas, args.num_personas)
    logger.info(f"Loaded {len(personas)} personas")
    
    # Initialize results
    results = {criterion: [] for criterion in criteria_keys}
    completed_questions = []
    
    # Check for checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.output_dir, args.model1, args.model2)
        if checkpoint:
            completed_questions = checkpoint.get("completed_questions", [])
            results = checkpoint.get("results", results)
            logger.info(f"Resuming from checkpoint: {len(completed_questions)} questions completed")
    
    completed_set = set(completed_questions)
    
    # Determine response columns
    response_col_1 = f"response_text_{args.model1}"
    response_col_2 = f"response_text_{args.model2}"
    
    # Try to find question text column
    question_col = None
    for col in ["question_text", f"question_text_{args.model1}", "question"]:
        if col in merged.columns:
            question_col = col
            break
    if question_col is None:
        raise ValueError("Could not find question text column in merged data")
    
    # Process questions
    try:
        for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Questions"):
            question_id = row["question_id"]
            
            if question_id in completed_set:
                continue
            
            question = row[question_col]
            response_1 = row[response_col_1]
            response_2 = row[response_col_2]
            
            # Evaluate with all personas
            evaluations = evaluate_question(
                personas, question, response_1, response_2,
                args.model1, args.model2, criteria_keys, config, 
                args.evaluator_model
            )
            
            # Calculate winners for each criterion
            for criterion in criteria_keys:
                result = calculate_winner(evaluations, criterion, args.model1, args.model2)
                results[criterion].append({
                    "question_id": question_id,
                    "question": question,
                    **result
                })
            
            completed_questions.append(question_id)
            
            # Checkpoint
            if len(completed_questions) % args.checkpoint_every == 0:
                save_checkpoint(args.output_dir, args.model1, args.model2, {
                    "completed_questions": completed_questions,
                    "results": results
                })
                logger.info(f"Checkpoint saved: {len(completed_questions)} questions")
    
    except Exception as e:
        if "QUOTA_EXCEEDED" in str(e):
            logger.error("API quota exceeded. Saving progress...")
        else:
            logger.error(f"Error: {e}")
        
        save_checkpoint(args.output_dir, args.model1, args.model2, {
            "completed_questions": completed_questions,
            "results": results
        })
        raise
    
    # Calculate summary statistics
    summary = {
        "model1": args.model1,
        "model2": args.model2,
        "total_questions": len(completed_questions),
        "num_personas": len(personas),
        "criteria": criteria_keys,
        "config_name": config.get("config_name", "unknown")
    }
    
    for criterion in criteria_keys:
        wins = Counter(r["winner"] for r in results[criterion])
        total = len(results[criterion])
        
        summary[criterion] = {
            f"{args.model1}_wins": wins[args.model1],
            f"{args.model2}_wins": wins[args.model2],
            "ties": wins["tie"],
            f"{args.model1}_winrate": (wins[args.model1] / total * 100) if total > 0 else 0,
            f"{args.model2}_winrate": (wins[args.model2] / total * 100) if total > 0 else 0
        }
    
    # Save results (including config used)
    save_results(args.output_dir, args.model1, args.model2, results, summary, config)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete: {args.model1} vs {args.model2}")
    print(f"{'='*60}")
    print(f"Config: {config.get('config_name', 'unknown')}")
    print(f"Total questions: {len(completed_questions)}")
    print(f"Personas: {len(personas)}")
    
    for criterion in criteria_keys:
        stats = summary[criterion]
        print(f"\n{criterion.upper()}:")
        print(f"  {args.model1}: {stats[f'{args.model1}_wins']} wins ({stats[f'{args.model1}_winrate']:.1f}%)")
        print(f"  {args.model2}: {stats[f'{args.model2}_wins']} wins ({stats[f'{args.model2}_winrate']:.1f}%)")
        print(f"  Ties: {stats['ties']}")


if __name__ == "__main__":
    main()
