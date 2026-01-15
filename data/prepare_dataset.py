"""
EPITOME Dataset Preparation for Therapeutic AI Training

This script processes the EPITOME corpus to extract unique therapeutic questions
(seeker posts) and creates train/test splits for model training and evaluation.

Reference:
    Sharma et al. (2020). "A Computational Approach to Understanding Empathy 
    Expressed in Text-Based Mental Health Support." EMNLP 2020.
    https://github.com/behavioral-data/Empathy-Mental-Health

Usage:
    python prepare_dataset.py --input_path /path/to/EPITOME.csv --output_dir ./processed
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_epitome_data(input_path: str) -> pd.DataFrame:
    """Load the EPITOME dataset from CSV file."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def extract_seeker_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and deduplicate seeker posts (therapeutic questions)."""
    print("\n=== Extracting Seeker Posts ===")
    
    # Extract relevant columns
    seeker_data = df[['sp_id', 'seeker_post']].copy()
    print(f"Total records: {len(seeker_data)}")
    
    # Remove rows with missing data
    seeker_data = seeker_data.dropna(subset=['sp_id', 'seeker_post'])
    print(f"After removing missing data: {len(seeker_data)}")
    
    # Remove duplicates (keep first occurrence of each sp_id)
    seeker_data = seeker_data.drop_duplicates(subset=['sp_id'], keep='first')
    print(f"After deduplication: {len(seeker_data)} unique posts")
    
    return seeker_data


def apply_quality_filters(
    df: pd.DataFrame,
    min_chars: int = 10,
    max_chars: int = 2000,
    min_words: int = 3
) -> pd.DataFrame:
    """Apply quality filters to ensure high-quality therapeutic questions."""
    print(f"\n=== Applying Quality Filters ===")
    print(f"Criteria: {min_chars}-{max_chars} chars, min {min_words} words")
    
    # Calculate text statistics
    df = df.copy()
    df['char_length'] = df['seeker_post'].str.len()
    df['word_count'] = df['seeker_post'].str.split().str.len()
    
    # Apply filters
    filtered = df[
        (df['char_length'] >= min_chars) &
        (df['char_length'] <= max_chars) &
        (df['word_count'] >= min_words)
    ].copy()
    
    print(f"After filtering: {len(filtered)} posts retained")
    print(f"Removed: {len(df) - len(filtered)} posts")
    
    # Drop helper columns
    filtered = filtered.drop(columns=['char_length', 'word_count'])
    
    return filtered


def create_train_test_split(
    df: pd.DataFrame,
    test_size: int = 600,
    random_state: int = 42,
    stratify_by_length: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified train/test split based on text length quartiles."""
    print(f"\n=== Creating Train/Test Split ===")
    
    # Rename columns to standard format
    dataset = df[['sp_id', 'seeker_post']].copy()
    dataset.columns = ['question_id', 'question_text']
    
    if stratify_by_length:
        # Create length quartiles for stratification
        dataset['length_quartile'] = pd.qcut(
            dataset['question_text'].str.len(),
            q=4,
            labels=['Q1_Short', 'Q2_Medium', 'Q3_Long', 'Q4_VeryLong']
        )
        
        train_data, test_data = train_test_split(
            dataset.drop('length_quartile', axis=1),
            test_size=test_size,
            random_state=random_state,
            stratify=dataset['length_quartile']
        )
    else:
        train_data, test_data = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state
        )
    
    print(f"Training set: {len(train_data)} questions")
    print(f"Test set: {len(test_data)} questions")
    print(f"Split ratio: {len(train_data)/len(dataset):.1%} / {len(test_data)/len(dataset):.1%}")
    
    # Verify mean lengths are similar (stratification check)
    train_mean = train_data['question_text'].str.len().mean()
    test_mean = test_data['question_text'].str.len().mean()
    print(f"Mean length - Train: {train_mean:.1f}, Test: {test_mean:.1f}")
    
    return train_data, test_data


def save_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str
) -> None:
    """Save processed datasets to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save files
    train_path = output_path / "questions_train.csv"
    test_path = output_path / "questions_test.csv"
    all_path = output_path / "questions_all.csv"
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Combine and save all data
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    all_data.to_csv(all_path, index=False)
    
    print(f"\n=== Files Saved ===")
    print(f"1. {train_path}")
    print(f"2. {test_path}")
    print(f"3. {all_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EPITOME dataset for therapeutic AI training"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to EPITOME.csv file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=600,
        help="Number of questions in test set"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=10,
        help="Minimum character length for questions"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=2000,
        help="Maximum character length for questions"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Minimum word count for questions"
    )
    
    args = parser.parse_args()
    
    # Process pipeline
    print("=" * 60)
    print("EPITOME Dataset Preparation for Therapeutic AI")
    print("=" * 60)
    
    # 1. Load data
    df = load_epitome_data(args.input_path)
    
    # 2. Extract seeker posts
    seeker_posts = extract_seeker_posts(df)
    
    # 3. Apply quality filters
    filtered_posts = apply_quality_filters(
        seeker_posts,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_words=args.min_words
    )
    
    # 4. Create train/test split
    train_data, test_data = create_train_test_split(
        filtered_posts,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # 5. Save datasets
    save_datasets(train_data, test_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
