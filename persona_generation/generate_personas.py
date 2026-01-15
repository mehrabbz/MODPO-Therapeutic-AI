"""
Patient Persona Generation Pipeline

This script processes survey data to generate patient personas for therapeutic AI evaluation.
It performs three main steps:
1. Generate personas from survey responses
2. Filter for quality and diversity
3. Split into train/test sets with demographic balance

PRIVACY NOTE:
    This script requires survey data files that are NOT included in this repository
    due to participant privacy. The expected input format is described below.

Input Format:
    Two CSV files with columns:
    - '1. What is your age group?'
    - '2. What is your gender?'
    - '3. How would you describe your ethnicity/race? (Check all that apply)'
    - '4. Have you ever been diagnosed...' (mental health diagnoses)
    - '5. How would you describe the severity...'
    - '6. Are you currently receiving treatment...'
    - '6. Have you ever interacted with an automated system...' (LLM experience)
    - '7a-7e' (AI attitudes - Likert scale)
    - Therapy preference columns (importance ratings 1-5)
    - '12. In your opinion, which areas...' (helpful areas)
    - '13. In which aspects of psychotherapy...' (appropriate uses)

Usage:
    python generate_personas.py \
        --survey_file_1 ./data/survey_group_1.csv \
        --survey_file_2 ./data/survey_group_2.csv \
        --output_dir ./personas \
        --num_personas 150 \
        --train_size 100 \
        --test_size 50

Output:
    - train_personas.json: Training set personas (100)
    - test_personas.json: Test set personas (50)
    - all_personas.json: Complete persona data
    - persona_report.txt: Generation and filtering report
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate patient personas from survey data"
    )
    parser.add_argument(
        "--survey_files",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to survey CSV file(s). Can provide one or two files."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Label(s) for survey group(s). Must match number of survey files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./personas",
        help="Output directory for persona files"
    )
    parser.add_argument(
        "--num_personas",
        type=int,
        default=150,
        help="Number of personas to select"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="Number of training personas"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=50,
        help="Number of test personas"
    )
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.9,
        help="Minimum quality score for persona selection"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


# =============================================================================
# Step 1: Persona Generation
# =============================================================================

def clean_response(response):
    """Clean and validate a survey response."""
    if pd.isna(response) or str(response).strip() == '':
        return None
    return str(response).strip()


def convert_attitude_to_text(attitude):
    """Convert Likert scale to natural language."""
    if not attitude:
        return None
    
    attitude_lower = attitude.lower()
    
    if 'strongly agree' in attitude_lower:
        return "strongly believe"
    elif 'agree' in attitude_lower and 'disagree' not in attitude_lower:
        return "think"
    elif 'neither' in attitude_lower or 'neutral' in attitude_lower:
        return "are unsure about whether"
    elif 'disagree' in attitude_lower and 'strongly' not in attitude_lower:
        return "don't think"
    elif 'strongly disagree' in attitude_lower:
        return "strongly disagree that"
    return attitude_lower


def convert_trust_level(trust):
    """Convert trust response to natural language."""
    if not trust:
        return None
    
    trust_lower = trust.lower().replace('trust trust', 'trust')
    
    if 'very high' in trust_lower or 'complete' in trust_lower:
        return "very high"
    elif 'high' in trust_lower:
        return "high"
    elif 'moderate' in trust_lower or 'medium' in trust_lower or 'neutral' in trust_lower:
        return "moderate"
    elif 'low' in trust_lower and 'very' not in trust_lower:
        return "low"
    elif 'very low' in trust_lower or 'no' in trust_lower:
        return "very low"
    return "moderate"


def generate_persona(participant, participant_index, population):
    """Generate a structured persona from participant data."""
    
    # Extract demographics
    age_group = clean_response(participant.get('1. What is your age group?'))
    gender = clean_response(participant.get('2. What is your gender?'))
    ethnicity = clean_response(participant.get('3. How would you describe your ethnicity/race? (Check all that apply)'))
    
    # Extract mental health info
    diagnoses = clean_response(participant.get('4.Have you ever been diagnosed by a healthcare provider with any of the following conditions (check all that apply)?'))
    severity = clean_response(participant.get('5. How would you describe the severity of your mental health condition?'))
    treatment = clean_response(participant.get('6. Are you currently receiving treatment for your mental health condition(s)? If yes, what type of treatment are you receiving? (Select all that apply)'))
    
    # Extract technology experience
    llm_experience = clean_response(participant.get('6.  Have you ever interacted with an automated system, such as a large language model (LLM) like ChatGPT or similar AI tools/applications, for any of the following purposes? (Check all that apply)'))
    
    # Extract AI attitudes
    accessible = clean_response(participant.get('7a. LLMs can help address mental health by offering accessible support.'))
    empathy = clean_response(participant.get('7b. LLMs can provide empathetic responses similar to human therapists.'))
    errors = clean_response(participant.get('7c. I am concerned about LLMs making factual errors in mental health conversations.'))
    complement = clean_response(participant.get('7d. I believe LLMs should only be used as a complement to human therapists, not as a replacement.'))
    trust = clean_response(participant.get('7e. To what extent do you trust LLMs to provide safe and accurate mental health support?'))
    
    # Extract therapy preferences
    empathy_score = clean_response(participant.get('Empathy in responses (showing warmth and compassion)'))
    listening_score = clean_response(participant.get('Active listening (understanding and reflecting the patient\'s feelings)'))
    change_score = clean_response(participant.get('Encouraging self-motivated change (helping patients articulate goals for change)'))
    trust_score = clean_response(participant.get('Building trust and rapport (creating a supportive connection)'))
    autonomy_score = clean_response(participant.get('Respecting patient autonomy (allowing patients to lead their own change journey)'))
    
    # Extract use cases
    helpful_areas = clean_response(participant.get('12. In your opinion, which areas of mental health do you think LLMs could be helpful? (Check all that apply'))
    appropriate_aspects = clean_response(participant.get('13. In which aspects of psychotherapy do you think it is appropriate to use LLMs? (Check all that apply)'))
    
    # Build persona text
    persona_lines = [
        "You are roleplaying as a patient with the following characteristics:",
        "",
        "DEMOGRAPHICS:"
    ]
    
    if age_group:
        persona_lines.append(f"• Age: {age_group}")
    if gender:
        persona_lines.append(f"• Gender: {gender}")
    if ethnicity:
        persona_lines.append(f"• Ethnicity: {ethnicity}")
    
    persona_lines.extend(["", "MENTAL HEALTH BACKGROUND:"])
    if diagnoses:
        persona_lines.append(f"• Diagnoses: {diagnoses}")
    if severity:
        persona_lines.append(f"• Severity level: {severity}")
    if treatment:
        persona_lines.append(f"• Current treatment: {treatment}")
    
    persona_lines.extend(["", "TECHNOLOGY EXPERIENCE:"])
    if llm_experience:
        persona_lines.append(f"You have previous experience with AI tools for:")
        for exp in str(llm_experience).split(','):
            exp = exp.strip()
            if exp:
                persona_lines.append(f"• {exp}")
    
    persona_lines.extend(["", "YOUR ATTITUDES TOWARD AI IN MENTAL HEALTH:"])
    if accessible:
        attitude = convert_attitude_to_text(accessible)
        persona_lines.append(f"• Accessibility: You {attitude} LLMs can help by offering accessible mental health support")
    if empathy:
        attitude = convert_attitude_to_text(empathy)
        persona_lines.append(f"• AI Empathy: You {attitude} LLMs can provide empathetic responses similar to human therapists")
    if errors:
        attitude = convert_attitude_to_text(errors)
        persona_lines.append(f"• Error Concerns: You {attitude} there should be much concern about LLMs making factual errors in mental health conversations")
    if complement:
        attitude = convert_attitude_to_text(complement)
        persona_lines.append(f"• Replacement vs Complement: You {attitude} LLMs should only complement, not replace, human therapists")
    if trust:
        trust_level = convert_trust_level(trust)
        persona_lines.append(f"• Trust Level: You have {trust_level} trust in LLMs for safe and accurate mental health support")
    
    persona_lines.extend(["", "THERAPY PREFERENCES (importance ratings out of 5):"])
    if empathy_score:
        persona_lines.append(f"• Empathy and warmth: {empathy_score}/5 importance")
    if listening_score:
        persona_lines.append(f"• Active listening: {listening_score}/5 importance")
    if trust_score:
        persona_lines.append(f"• Trust and rapport building: {trust_score}/5 importance")
    if change_score:
        persona_lines.append(f"• Self-motivated change support: {change_score}/5 importance")
    if autonomy_score:
        persona_lines.append(f"• Patient autonomy: {autonomy_score}/5 importance")
    
    if helpful_areas:
        persona_lines.extend(["", "AREAS WHERE YOU THINK AI COULD BE HELPFUL:"])
        for area in str(helpful_areas).split(','):
            area = area.strip()
            if area:
                persona_lines.append(f"• {area}")
    
    if appropriate_aspects:
        persona_lines.extend(["", "YOUR VIEWS ON APPROPRIATE USES OF AI IN THERAPY:"])
        persona_lines.append("You think AI is appropriate for:")
        for aspect in str(appropriate_aspects).split(','):
            aspect = aspect.strip()
            if aspect:
                persona_lines.append(f"• {aspect}")
    
    # Create structured data
    persona_data = {
        "participant_index": participant_index,
        "population": population,
        "demographics": {
            "age_group": age_group,
            "gender": gender,
            "ethnicity": ethnicity
        },
        "mental_health": {
            "diagnoses": diagnoses,
            "severity": severity,
            "treatment": treatment
        },
        "therapy_preferences": {
            "empathy_warmth": empathy_score,
            "active_listening": listening_score,
            "self_motivated_change": change_score,
            "trust_rapport": trust_score,
            "patient_autonomy": autonomy_score
        },
        "persona_text": "\n".join(persona_lines)
    }
    
    return persona_data


def generate_all_personas(df, population_label):
    """Generate personas for all participants in a dataframe."""
    personas = []
    
    for i in range(len(df)):
        try:
            persona = generate_persona(df.iloc[i], i, population_label)
            personas.append(persona)
        except Exception as e:
            print(f"Warning: Failed to generate persona {i}: {e}")
            continue
    
    return personas


# =============================================================================
# Step 2: Quality Filtering and Selection
# =============================================================================

def calculate_quality_score(persona_text):
    """Calculate quality score for a persona."""
    score = 0
    
    # Completeness: Check for required sections
    required_sections = ['DEMOGRAPHICS', 'MENTAL HEALTH BACKGROUND', 
                        'TECHNOLOGY EXPERIENCE', 'ATTITUDES TOWARD AI', 
                        'THERAPY PREFERENCES']
    sections_found = sum(1 for s in required_sections if s in persona_text)
    completeness = sections_found / len(required_sections)
    score += completeness * 0.4
    
    # Consistency: Check for logical responses
    consistency = 1.0
    if 'No reported diagnoses' in persona_text and 'Severe' in persona_text:
        consistency -= 0.4
    score += consistency * 0.3
    
    # Specificity: Check for detailed content
    specific_terms = ['therapy', 'counseling', 'medication', 'anxiety', 
                     'depression', 'ChatGPT', 'support']
    specific_count = sum(1 for t in specific_terms if t.lower() in persona_text.lower())
    specificity = min(1.0, specific_count / 5)
    score += specificity * 0.2
    
    # Validity: Check basic formatting
    validity = 1.0
    if persona_text.count('•') < 3:
        validity -= 0.2
    if 'Age:' not in persona_text:
        validity -= 0.2
    score += max(0, validity) * 0.1
    
    return score


def compute_similarity_matrix(personas):
    """Compute pairwise similarity between personas."""
    texts = [p['persona_text'] for p in personas]
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix)
    return similarity


def select_diverse_personas(personas, target_count, min_quality):
    """Select diverse, high-quality personas."""
    print(f"Selecting {target_count} diverse personas...")
    
    # Calculate quality scores
    quality_scores = []
    for p in personas:
        score = calculate_quality_score(p['persona_text'])
        quality_scores.append(score)
    
    # Filter by quality
    high_quality_indices = [i for i, s in enumerate(quality_scores) if s >= min_quality]
    print(f"Found {len(high_quality_indices)} personas with quality >= {min_quality}")
    
    if len(high_quality_indices) <= target_count:
        return [personas[i] for i in high_quality_indices], quality_scores
    
    # Compute similarity
    high_quality_personas = [personas[i] for i in high_quality_indices]
    similarity = compute_similarity_matrix(high_quality_personas)
    
    # Greedy diverse selection
    selected_indices = []
    
    # Start with highest quality
    hq_scores = [quality_scores[i] for i in high_quality_indices]
    best_idx = np.argmax(hq_scores)
    selected_indices.append(best_idx)
    
    # Add most dissimilar personas
    while len(selected_indices) < target_count:
        max_min_dist = -1
        best_candidate = None
        
        for i in range(len(high_quality_personas)):
            if i in selected_indices:
                continue
            
            min_dist = min(1 - similarity[i][j] for j in selected_indices)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = i
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
        else:
            break
    
    selected_personas = [high_quality_personas[i] for i in selected_indices]
    return selected_personas, quality_scores


# =============================================================================
# Step 3: Train/Test Split with Demographic Balance
# =============================================================================

def get_primary_ethnicity(ethnicity_str):
    """Assign primary ethnicity category."""
    if not ethnicity_str:
        return 'Other'
    
    eth_lower = str(ethnicity_str).lower()
    
    if 'hispanic' in eth_lower or 'latino' in eth_lower:
        return 'Hispanic'
    elif 'middle eastern' in eth_lower or 'north african' in eth_lower:
        return 'MiddleEastern'
    elif 'black' in eth_lower or 'african american' in eth_lower:
        return 'Black'
    elif 'white' in eth_lower or 'caucasian' in eth_lower:
        return 'White'
    elif 'asian' in eth_lower:
        return 'Asian'
    return 'Other'


def create_balanced_split(personas, train_size, test_size, random_state, stratify_by_population=True):
    """Create demographically balanced train/test split."""
    print(f"Creating balanced split: {train_size} train, {test_size} test")
    
    np.random.seed(random_state)
    
    # Extract demographics
    for p in personas:
        p['_ethnicity'] = get_primary_ethnicity(p['demographics'].get('ethnicity'))
        p['_gender'] = p['demographics'].get('gender', 'Unknown')
        p['_population'] = p.get('population', 'Unknown')
    
    # Count demographics
    ethnicity_counts = Counter(p['_ethnicity'] for p in personas)
    population_counts = Counter(p['_population'] for p in personas)
    print(f"Ethnicity distribution: {dict(ethnicity_counts)}")
    print(f"Population distribution: {dict(population_counts)}")
    
    # Check if stratification by population is meaningful
    unique_populations = list(population_counts.keys())
    if len(unique_populations) <= 1:
        stratify_by_population = False
        print("Single population detected - skipping population stratification")
    
    # Select test set with demographic balance
    test_personas = []
    remaining = personas.copy()
    
    # Ensure minority ethnicity representation (e.g., Middle Eastern)
    for ethnicity in ['MiddleEastern', 'Hispanic']:
        candidates = [p for p in remaining if p['_ethnicity'] == ethnicity]
        if candidates:
            n_select = min(2, len(candidates))
            selected = np.random.choice(len(candidates), n_select, replace=False)
            for idx in selected:
                test_personas.append(candidates[idx])
                remaining.remove(candidates[idx])
    
    # Fill remaining test slots
    if stratify_by_population:
        # Balance by population
        while len(test_personas) < test_size and remaining:
            for pop in unique_populations:
                if len(test_personas) >= test_size:
                    break
                candidates = [p for p in remaining if p['_population'] == pop]
                if candidates:
                    selected = np.random.choice(len(candidates), 1)[0]
                    test_personas.append(candidates[selected])
                    remaining.remove(candidates[selected])
    else:
        # Random selection without population stratification
        while len(test_personas) < test_size and remaining:
            selected = np.random.choice(len(remaining), 1)[0]
            test_personas.append(remaining[selected])
            remaining.remove(remaining[selected])
    
    train_personas = remaining[:train_size]
    
    # Clean up temporary fields
    for p in personas:
        del p['_ethnicity'], p['_gender'], p['_population']
    
    return train_personas, test_personas


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Patient Persona Generation Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate arguments
    if args.labels and len(args.labels) != len(args.survey_files):
        print("Error: Number of labels must match number of survey files")
        return
    
    # Default labels if not provided
    if args.labels:
        labels = args.labels
    else:
        labels = [f"Group{i+1}" for i in range(len(args.survey_files))]
    
    # Load survey data
    print(f"\nLoading survey data...")
    dataframes = []
    total_participants = 0
    
    for survey_file, label in zip(args.survey_files, labels):
        try:
            df = pd.read_csv(survey_file)
            df['Population'] = label
            dataframes.append(df)
            print(f"Loaded {len(df)} participants from {label} ({survey_file})")
            total_participants += len(df)
        except FileNotFoundError as e:
            print(f"Error: Could not find survey file: {e}")
            return
    
    print(f"Total: {total_participants} participants")
    
    # Combine dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Check if we have multiple populations for stratification
    has_multiple_populations = len(args.survey_files) > 1
    
    # Step 1: Generate all personas
    print(f"\n--- Step 1: Generating Personas ---")
    all_personas = []
    for df, label in zip(dataframes, labels):
        personas = generate_all_personas(df, label)
        all_personas.extend(personas)
    print(f"Generated {len(all_personas)} total personas")
    
    # Step 2: Filter and select
    print(f"\n--- Step 2: Filtering and Selection ---")
    selected_personas, quality_scores = select_diverse_personas(
        all_personas, 
        args.num_personas, 
        args.min_quality
    )
    print(f"Selected {len(selected_personas)} personas")
    
    # Step 3: Create train/test split
    print(f"\n--- Step 3: Train/Test Split ---")
    train_personas, test_personas = create_balanced_split(
        selected_personas,
        args.train_size,
        args.test_size,
        args.random_state,
        stratify_by_population=has_multiple_populations
    )
    print(f"Train set: {len(train_personas)} personas")
    print(f"Test set: {len(test_personas)} personas")
    
    # Save outputs
    print(f"\n--- Saving Outputs ---")
    
    # Save train personas
    train_output = {f"persona_{i:03d}": p['persona_text'] for i, p in enumerate(train_personas)}
    with open(output_dir / "train_personas.json", 'w', encoding='utf-8') as f:
        json.dump(train_output, f, indent=2, ensure_ascii=False)
    
    # Save test personas
    test_output = {f"persona_{i:03d}": p['persona_text'] for i, p in enumerate(test_personas)}
    with open(output_dir / "test_personas.json", 'w', encoding='utf-8') as f:
        json.dump(test_output, f, indent=2, ensure_ascii=False)
    
    # Save complete data (for analysis)
    all_output = {
        "metadata": {
            "generation_date": datetime.now().isoformat(),
            "total_generated": len(all_personas),
            "total_selected": len(selected_personas),
            "train_count": len(train_personas),
            "test_count": len(test_personas)
        },
        "train_personas": train_personas,
        "test_personas": test_personas
    }
    with open(output_dir / "all_personas.json", 'w', encoding='utf-8') as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)
    
    # Generate report
    input_lines = [f"  {label}: {len(df)} participants" 
                   for df, label in zip(dataframes, labels)]
    
    report_lines = [
        "PERSONA GENERATION REPORT",
        "=" * 50,
        f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUT:",
        *input_lines,
        f"  Total: {total_participants} participants",
        "",
        "PROCESSING:",
        f"  Generated personas: {len(all_personas)}",
        f"  Min quality threshold: {args.min_quality}",
        f"  Selected personas: {len(selected_personas)}",
        f"  Population stratification: {'Yes' if has_multiple_populations else 'No'}",
        "",
        "OUTPUT:",
        f"  Training personas: {len(train_personas)}",
        f"  Test personas: {len(test_personas)}",
        "",
        "QUALITY STATISTICS:",
        f"  Mean quality score: {np.mean(quality_scores):.3f}",
        f"  Std deviation: {np.std(quality_scores):.3f}",
        f"  Min score: {np.min(quality_scores):.3f}",
        f"  Max score: {np.max(quality_scores):.3f}",
        "",
        "FILES CREATED:",
        "  • train_personas.json",
        "  • test_personas.json", 
        "  • all_personas.json",
        "  • persona_report.txt"
    ]
    
    with open(output_dir / "persona_report.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nFiles saved to: {output_dir}/")
    print("  • train_personas.json")
    print("  • test_personas.json")
    print("  • all_personas.json")
    print("  • persona_report.txt")
    
    print("\n" + "=" * 60)
    print("Persona generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
