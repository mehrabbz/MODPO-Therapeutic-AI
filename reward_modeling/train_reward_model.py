"""
Reward Model Training

Train RoBERTa-based reward models on preference data for therapeutic AI alignment.
Uses Optuna for hyperparameter optimization and TRL's RewardTrainer.

Usage:
    python train_reward_model.py \
        --preferences ./preferences/empathy_pairs.json \
        --output_dir ./reward_models/empathy \
        --criterion empathy

    # With hyperparameter search
    python train_reward_model.py \
        --preferences ./preferences/empathy_pairs.json \
        --output_dir ./reward_models/empathy \
        --criterion empathy \
        --n_trials 20

    # Skip hyperparameter search (use defaults or provided values)
    python train_reward_model.py \
        --preferences ./preferences/empathy_pairs.json \
        --output_dir ./reward_models/empathy \
        --criterion empathy \
        --skip_hpo
"""

import argparse
import gc
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train reward models on preference data"
    )
    parser.add_argument(
        "--preferences",
        type=str,
        required=True,
        help="Path to preference pairs JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        required=True,
        help="Name of the criterion (for logging)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help="Base model for reward model"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials for HPO"
    )
    parser.add_argument(
        "--skip_hpo",
        action="store_true",
        help="Skip hyperparameter optimization"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (used if --skip_hpo)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.03,
        help="Weight decay (used if --skip_hpo)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (used if --skip_hpo)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (used if --skip_hpo)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (used if --skip_hpo)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


def load_preference_data(file_path: str) -> list[dict]:
    """
    Load preference pairs from JSON file.
    
    Expected format (from collect_preferences.py):
    [
        {
            "question_id": "...",
            "question_text": "...",
            "chosen": {"response_id": ..., "response_text": "..."},
            "rejected": {"response_id": ..., "response_text": "..."}
        }
    ]
    """
    logger.info(f"Loading preferences from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to training format
    formatted_data = []
    for item in data:
        formatted_data.append({
            "prompt": item["question_text"],
            "chosen": item["chosen"]["response_text"],
            "rejected": item["rejected"]["response_text"],
            "question_id": str(item["question_id"])
        })
    
    logger.info(f"Loaded {len(formatted_data)} preference pairs")
    return formatted_data


def prepare_dataset_for_trl(
    data: list[dict],
    tokenizer,
    max_length: int = 512
) -> Dataset:
    """Prepare dataset in format expected by TRL RewardTrainer."""
    
    # Tokenize prompts
    prompts = [item["prompt"] for item in data]
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length // 2,
        padding=False
    )
    
    processed_items = []
    
    for i, item in enumerate(data):
        prompt_input_ids = tokenized_prompts["input_ids"][i]
        prompt_attention_mask = tokenized_prompts["attention_mask"][i]
        
        # Tokenize chosen response
        chosen_tokenized = tokenizer(
            item["chosen"],
            truncation=True,
            max_length=max_length - len(prompt_input_ids),
            padding=False
        )
        
        # Tokenize rejected response
        rejected_tokenized = tokenizer(
            item["rejected"],
            truncation=True,
            max_length=max_length - len(prompt_input_ids),
            padding=False
        )
        
        # Combine prompt + response
        chosen_input_ids = prompt_input_ids + chosen_tokenized["input_ids"]
        chosen_attention_mask = prompt_attention_mask + chosen_tokenized["attention_mask"]
        rejected_input_ids = prompt_input_ids + rejected_tokenized["input_ids"]
        rejected_attention_mask = prompt_attention_mask + rejected_tokenized["attention_mask"]
        
        # Pad to max_length
        pad_token_id = tokenizer.pad_token_id
        
        chosen_input_ids = (chosen_input_ids + [pad_token_id] * (max_length - len(chosen_input_ids)))[:max_length]
        chosen_attention_mask = (chosen_attention_mask + [0] * (max_length - len(chosen_attention_mask)))[:max_length]
        rejected_input_ids = (rejected_input_ids + [pad_token_id] * (max_length - len(rejected_input_ids)))[:max_length]
        rejected_attention_mask = (rejected_attention_mask + [0] * (max_length - len(rejected_attention_mask)))[:max_length]
        
        processed_items.append({
            "input_ids_chosen": chosen_input_ids,
            "attention_mask_chosen": chosen_attention_mask,
            "input_ids_rejected": rejected_input_ids,
            "attention_mask_rejected": rejected_attention_mask
        })
    
    return Dataset.from_list(processed_items)


def calculate_accuracy(
    model,
    val_dataset: Dataset,
    device: torch.device,
    batch_size: int = 8
) -> float:
    """Calculate validation accuracy (chosen reward > rejected reward)."""
    
    model.eval()
    correct = 0
    total = 0
    
    for i in range(0, len(val_dataset), batch_size):
        end = min(i + batch_size, len(val_dataset))
        
        chosen_input_ids = torch.tensor(
            [val_dataset[j]["input_ids_chosen"] for j in range(i, end)]
        ).to(device)
        chosen_attention_mask = torch.tensor(
            [val_dataset[j]["attention_mask_chosen"] for j in range(i, end)]
        ).to(device)
        rejected_input_ids = torch.tensor(
            [val_dataset[j]["input_ids_rejected"] for j in range(i, end)]
        ).to(device)
        rejected_attention_mask = torch.tensor(
            [val_dataset[j]["attention_mask_rejected"] for j in range(i, end)]
        ).to(device)
        
        with torch.no_grad():
            chosen_rewards = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            ).logits.squeeze(-1)
            rejected_rewards = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            ).logits.squeeze(-1)
        
        predictions = (chosen_rewards > rejected_rewards).cpu().numpy()
        correct += sum(predictions)
        total += len(predictions)
        
        # Clean up
        del chosen_input_ids, chosen_attention_mask
        del rejected_input_ids, rejected_attention_mask
        del chosen_rewards, rejected_rewards
        torch.cuda.empty_cache()
    
    return correct / total if total > 0 else 0.0


def train_reward_model(
    train_data: list[dict],
    val_data: list[dict] = None,
    model_name: str = "roberta-large",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.03,
    dropout: float = 0.1,
    batch_size: int = 8,
    max_length: int = 512,
    num_epochs: int = 5,
    output_dir: str = None,
    device: torch.device = None
) -> tuple[str, dict]:
    """
    Train a reward model on preference data.
    
    Returns:
        Tuple of (model_path, training_info)
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    save_model = output_dir is not None
    if save_model:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare datasets
    train_dataset = prepare_dataset_for_trl(train_data, tokenizer, max_length)
    val_dataset = prepare_dataset_for_trl(val_data, tokenizer, max_length) if val_data else None
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    model.to(device)
    
    # Training config
    training_args = RewardConfig(
        output_dir=output_dir if save_model else "./tmp_reward",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        max_length=max_length,
        disable_dropout=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        report_to=[]
    )
    
    # Train
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    accuracy = 0.0
    if val_dataset:
        accuracy = calculate_accuracy(model, val_dataset, device, batch_size)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = None
    if save_model:
        model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")
    
    training_info = {
        "accuracy": accuracy,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "max_length": max_length
    }
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return model_path, training_info


def run_hpo(
    train_data: list[dict],
    val_data: list[dict],
    n_trials: int,
    device: torch.device
) -> dict:
    """Run hyperparameter optimization using Optuna."""
    
    def objective(trial):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 5e-7, 2e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.15, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        max_length = trial.suggest_categorical("max_length", [256, 512])
        num_epochs = trial.suggest_int("num_epochs", 3, 7)
        
        try:
            _, training_info = train_reward_model(
                train_data=train_data,
                val_data=val_data,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                batch_size=batch_size,
                max_length=max_length,
                num_epochs=num_epochs,
                output_dir=None,
                device=device
            )
            return training_info["accuracy"]
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return 0.0
    
    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best trial accuracy: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")
    
    return study.best_trial.params


def main():
    args = parse_args()
    
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set random seeds
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"Reward Model Training: {args.criterion}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    
    # Load data
    preference_data = load_preference_data(args.preferences)
    
    # Shuffle and split
    np.random.shuffle(preference_data)
    val_size = int(len(preference_data) * args.val_split)
    train_data = preference_data[val_size:]
    val_data = preference_data[:val_size]
    
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Get hyperparameters
    if args.skip_hpo:
        logger.info("Skipping HPO, using provided/default hyperparameters")
        best_params = {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_epochs": args.num_epochs
        }
    else:
        logger.info(f"Running HPO with {args.n_trials} trials...")
        best_params = run_hpo(train_data, val_data, args.n_trials, device)
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_dir = output_dir / f"{args.criterion}_{timestamp}"
    
    model_path, final_info = train_reward_model(
        train_data=preference_data,  # Use all data for final model
        val_data=None,
        model_name=args.model_name,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"],
        max_length=best_params["max_length"],
        num_epochs=best_params["num_epochs"],
        output_dir=str(final_output_dir),
        device=device
    )
    
    # Save training info
    info_path = final_output_dir / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            "criterion": args.criterion,
            "model_name": args.model_name,
            "best_params": best_params,
            "final_training_info": final_info,
            "data_size": len(preference_data),
            "timestamp": timestamp
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {final_output_dir}")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
