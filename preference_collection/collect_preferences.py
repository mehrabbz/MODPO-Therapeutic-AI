"""
Preference Collection using Patient Personas

This script collects preferences by having LLM-based patient personas evaluate
and rank therapeutic responses according to configurable criteria.

The criteria are loaded from a JSON config file, making it easy to:
- Use different evaluation frameworks (therapeutic, Gricean maxims, safety, etc.)
- Add or modify criteria without changing code
- Run evaluations with different criteria configurations

Usage:
    python collect_preferences.py \
        --config ./configs/therapeutic_criteria.json \
        --personas ./personas/train_personas.json \
        --responses ./responses/responses_train.csv \
        --output_dir ./preferences \
        --personas_per_question 50

Output:
    For each criterion in the config:
    - {criterion}_pairs.json: Best/worst preference pairs for DPO training
    - {criterion}_rankings.json: Complete rankings for analysis
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect preferences using patient personas"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to criteria config JSON file"
    )
    parser.add_argument(
        "--personas",
        type=str,
        required=True,
        help="Path to personas JSON file"
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to responses CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preferences",
        help="Output directory for preference data"
    )
    parser.add_argument(
        "--personas_per_question",
        type=int,
        default=50,
        help="Number of personas to sample per question"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for evaluation"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=10,
        help="Save checkpoint every N questions"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load evaluation criteria from config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Loaded config: {config['config_name']}")
    print(f"Description: {config['description']}")
    print(f"Criteria: {list(config['criteria'].keys())}")
    
    return config


def load_personas(personas_path: str) -> dict:
    """Load patient personas from JSON file."""
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    print(f"Loaded {len(personas)} personas")
    return personas


def load_responses(responses_path: str) -> pd.DataFrame:
    """Load responses from CSV file."""
    df = pd.read_csv(responses_path)
    print(f"Loaded {len(df)} responses")
    print(f"Unique questions: {df['question_id'].nunique()}")
    return df


def build_evaluation_prompt(
    persona_text: str,
    question: str,
    responses: list[str],
    criteria: dict,
    num_responses: int
) -> str:
    """
    Build the evaluation prompt dynamically from config criteria.
    
    Args:
        persona_text: The patient persona description
        question: The therapeutic question
        responses: List of responses to evaluate
        criteria: Dictionary of criteria from config
        num_responses: Number of responses being ranked
    
    Returns:
        Complete prompt for LLM evaluation
    """
    # Format responses
    formatted_responses = "\n\n".join([
        f"Response {i+1}:\n{response}" 
        for i, response in enumerate(responses)
    ])
    
    # Build criteria descriptions
    criteria_descriptions = []
    for i, (key, criterion) in enumerate(criteria.items(), 1):
        desc = f"""{i}. **{criterion['name']}** – {criterion['description']}
   - Better ranks when: {criterion['better_when']}
   - Worse ranks when: {criterion['worse_when']}"""
        criteria_descriptions.append(desc)
    
    criteria_text = "\n\n".join(criteria_descriptions)
    
    # Build expected JSON fields
    json_fields = []
    for key in criteria.keys():
        json_fields.append(f'"{key}_ranking": [Response numbers from best to worst]')
    json_fields.append('"explanation": "Brief explanation of your rankings"')
    json_template = "{\n  " + ",\n  ".join(json_fields) + "\n}"
    
    # Build ranking instruction
    ranking_range = f"from best (1) to worst ({num_responses})"
    
    prompt = f"""{persona_text}

EVALUATION TASK:
Imagine you are in a therapy session and you asked the following question:

"{question}"

You received these different responses from various therapists:

{formatted_responses}

Based on your characteristics and preferences described above, please evaluate these responses on the following criteria:

{criteria_text}

Please rank the {num_responses} responses for each criterion {ranking_range}, and explain your reasoning. Use the following JSON format in your reply:

{json_template}
"""
    
    return prompt


def get_evaluation(
    client: OpenAI,
    prompt: str,
    criteria_keys: list[str],
    model: str = "gpt-4o",
    max_retries: int = 3
) -> dict:
    """
    Call OpenAI API to get evaluation rankings.
    
    Args:
        client: OpenAI client
        prompt: The evaluation prompt
        criteria_keys: List of criteria keys to validate
        model: Model to use
        max_retries: Number of retry attempts
    
    Returns:
        Parsed evaluation dictionary
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            evaluation = json.loads(response_text)
            
            # Validate required fields
            required_fields = [f"{key}_ranking" for key in criteria_keys]
            required_fields.append("explanation")
            
            if all(field in evaluation for field in required_fields):
                return evaluation
            else:
                missing = [f for f in required_fields if f not in evaluation]
                print(f"Missing fields: {missing}. Retrying...")
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}. Retrying...")
        except Exception as e:
            print(f"API error: {e}. Retrying in {2 ** attempt}s...")
            time.sleep(2 ** attempt)
    
    raise Exception(f"Failed after {max_retries} attempts")


def evaluate_with_personas(
    client: OpenAI,
    personas: dict,
    question: str,
    responses: list[str],
    criteria: dict,
    num_responses: int,
    sample_size: int,
    model: str,
    random_state: int
) -> dict:
    """
    Get evaluations from a sample of patient personas.
    
    Args:
        client: OpenAI client
        personas: Dictionary of personas
        question: The question to evaluate
        responses: List of responses
        criteria: Criteria configuration
        num_responses: Number of responses
        sample_size: Number of personas to sample
        model: Model to use
        random_state: Random seed
    
    Returns:
        Dictionary of evaluations by persona ID
    """
    random.seed(random_state)
    
    persona_ids = list(personas.keys())
    if sample_size < len(persona_ids):
        persona_ids = random.sample(persona_ids, sample_size)
    
    criteria_keys = list(criteria.keys())
    evaluations = {}
    
    for persona_id in tqdm(persona_ids, desc="Evaluating personas", leave=False):
        persona_text = personas[persona_id]
        prompt = build_evaluation_prompt(
            persona_text, question, responses, criteria, num_responses
        )
        
        try:
            evaluation = get_evaluation(client, prompt, criteria_keys, model)
            evaluations[persona_id] = evaluation
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error with {persona_id}: {e}")
    
    return evaluations


def symmetrical_weighted_voting(
    evaluations: dict,
    ranking_key: str,
    num_responses: int,
    weights: list[int]
) -> list[int]:
    """
    Aggregate rankings using symmetrical weighted voting.
    
    Args:
        evaluations: Dictionary of evaluations by persona
        ranking_key: Key for the ranking to aggregate (e.g., "empathy_ranking")
        num_responses: Number of responses being ranked
        weights: List of weights for each position
    
    Returns:
        List of response numbers from best to worst
    """
    response_scores = {i: 0 for i in range(1, num_responses + 1)}
    
    for persona_id, evaluation in evaluations.items():
        rankings = evaluation.get(ranking_key, [])
        
        # Validate rankings
        if not isinstance(rankings, list) or len(rankings) != num_responses:
            continue
        if len(set(rankings)) != num_responses:
            continue
        if not all(1 <= r <= num_responses for r in rankings):
            continue
        
        # Apply weights
        for position, response_num in enumerate(rankings):
            response_scores[response_num] += weights[position]
    
    # Sort by score (highest first)
    sorted_responses = sorted(
        response_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [r for r, _ in sorted_responses]


def ensure_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {ensure_serializable(k): ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(i) for i in obj]
    return obj


def save_results(
    output_dir: Path,
    criteria_keys: list[str],
    all_pairs: dict,
    all_rankings: dict
):
    """Save preference pairs and rankings to JSON files."""
    for key in criteria_keys:
        # Save pairs
        pairs_path = output_dir / f"{key}_pairs.json"
        with open(pairs_path, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(all_pairs[key]), f, indent=2)
        
        # Save rankings
        rankings_path = output_dir / f"{key}_rankings.json"
        with open(rankings_path, 'w', encoding='utf-8') as f:
            json.dump(ensure_serializable(all_rankings[key]), f, indent=2)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Preference Collection with Patient Personas")
    print("=" * 60)
    
    # Set random seed
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Load config and data
    config = load_config(args.config)
    personas = load_personas(args.personas)
    responses_df = load_responses(args.responses)
    
    # Setup
    criteria = config['criteria']
    criteria_keys = list(criteria.keys())
    num_responses = config['num_responses_to_rank']
    weights = config['voting_weights'][f'weights_{num_responses}_responses']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Initialize results containers
    all_pairs = {key: [] for key in criteria_keys}
    all_rankings = {key: [] for key in criteria_keys}
    
    # Group responses by question
    question_groups = responses_df.groupby('question_id')
    
    print(f"\nProcessing {len(question_groups)} questions...")
    print(f"Personas per question: {args.personas_per_question}")
    print(f"Criteria: {criteria_keys}")
    
    # Process each question
    for q_idx, (question_id, group) in enumerate(tqdm(question_groups, desc="Questions")):
        question_text = group['question_text'].iloc[0]
        responses_list = group['response_text'].tolist()[:num_responses]
        response_ids = group['response_id'].tolist()[:num_responses]
        
        if len(responses_list) < num_responses:
            print(f"Warning: Question {question_id} has only {len(responses_list)} responses")
            continue
        
        # Get evaluations from personas
        evaluations = evaluate_with_personas(
            client=client,
            personas=personas,
            question=question_text,
            responses=responses_list,
            criteria=criteria,
            num_responses=num_responses,
            sample_size=args.personas_per_question,
            model=args.model,
            random_state=args.random_state + q_idx
        )
        
        # Aggregate rankings for each criterion
        for key in criteria_keys:
            ranking_key = f"{key}_ranking"
            final_ranking = symmetrical_weighted_voting(
                evaluations, ranking_key, num_responses, weights
            )
            
            # Create preference pair (best vs worst)
            best_idx = final_ranking[0] - 1
            worst_idx = final_ranking[-1] - 1
            
            pair = {
                "question_id": ensure_serializable(question_id),
                "question_text": question_text,
                "chosen": {
                    "response_id": ensure_serializable(response_ids[best_idx]),
                    "response_text": responses_list[best_idx]
                },
                "rejected": {
                    "response_id": ensure_serializable(response_ids[worst_idx]),
                    "response_text": responses_list[worst_idx]
                }
            }
            all_pairs[key].append(pair)
            
            # Create complete ranking
            ranking_entry = {
                "question_id": ensure_serializable(question_id),
                "question_text": question_text,
                "ranking": [
                    {
                        "rank": i + 1,
                        "response_id": ensure_serializable(response_ids[r - 1]),
                        "response_text": responses_list[r - 1]
                    }
                    for i, r in enumerate(final_ranking)
                ]
            }
            all_rankings[key].append(ranking_entry)
        
        # Periodic save
        if (q_idx + 1) % args.save_frequency == 0:
            print(f"\nSaving checkpoint at question {q_idx + 1}...")
            save_results(output_dir, criteria_keys, all_pairs, all_rankings)
    
    # Final save
    save_results(output_dir, criteria_keys, all_pairs, all_rankings)
    
    # Summary
    print("\n" + "=" * 60)
    print("Preference Collection Complete!")
    print("=" * 60)
    print(f"Questions processed: {len(question_groups)}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    for key in criteria_keys:
        print(f"  • {key}_pairs.json ({len(all_pairs[key])} pairs)")
        print(f"  • {key}_rankings.json ({len(all_rankings[key])} rankings)")


if __name__ == "__main__":
    main()
