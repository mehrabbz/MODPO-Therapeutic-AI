"""
MODPO (Multi-Objective Direct Preference Optimization) Trainer

Extends the DPOTrainer to support training with multiple objectives by incorporating
margin reward values from auxiliary reward models into the loss function.

Reference: Zhou et al., "Beyond One-Preference-Fits-All Alignment: Multi-Objective 
Direct Preference Optimization" (ACL 2024)
"""

import torch
import torch.nn.functional as F
from trl import DPOTrainer


class MODPOTrainer(DPOTrainer):
    """
    Multi-Objective Direct Preference Optimization (MODPO) Trainer.
    
    Extends DPOTrainer to support training with multiple objectives by incorporating
    margin rewards from auxiliary reward models directly into the loss function.
    
    The MODPO loss incorporates a margin term computed from auxiliary reward models:
    
    L_MODPO = -log σ(β/w_k * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)) 
                     - 1/w_k * margin)
    
    where margin = Σ_{j≠k} w_j * (r_j(x, y_w) - r_j(x, y_l))
    
    Args:
        model: The model to train
        ref_model: Reference model (optional, uses model with LoRA disabled if None)
        weights: List of weights for each objective [w_primary, w_margin1, ...]
        num_objectives: Total number of objectives (primary + margin objectives)
        **kwargs: Additional arguments for DPOTrainer
    """
    
    def __init__(
        self,
        model,
        ref_model=None,
        weights=None,
        num_objectives=2,
        **kwargs
    ):
        super().__init__(model=model, ref_model=ref_model, **kwargs)
        
        # Get beta from args
        self.beta = self.args.beta if hasattr(self.args, "beta") else 0.1
        
        # Set weights for objectives
        if weights is None:
            weight_value = 1.0 / num_objectives
            self.weights = torch.tensor(
                [weight_value] * num_objectives,
                device=self.accelerator.device
            )
        else:
            self.weights = torch.tensor(weights, device=self.accelerator.device)
        
        print(f"MODPO initialized with weights: {self.weights.tolist()} and beta: {self.beta}")
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        margin_values: torch.FloatTensor = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the MODPO loss for a batch.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy
            policy_rejected_logps: Log probs of rejected responses under policy
            reference_chosen_logps: Log probs of chosen responses under reference
            reference_rejected_logps: Log probs of rejected responses under reference
            margin_values: Pre-computed margin values from auxiliary reward models
            
        Returns:
            losses: Per-example losses
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
        """
        # Compute log ratios
        policy_logps_diff = policy_chosen_logps - policy_rejected_logps
        reference_logps_diff = reference_chosen_logps - reference_rejected_logps
        
        # Standard DPO logits
        logits = policy_logps_diff - reference_logps_diff
        
        # Apply MODPO scaling with primary weight
        w_k = self.weights[0]  # Primary objective weight
        scaled_logits = (self.beta / w_k) * logits
        
        # Add margin term if provided
        if margin_values is not None:
            # margin_values should already be: Σ_{j≠k} w_j * (r_j(y_w) - r_j(y_l))
            margin_term = (1.0 / w_k) * margin_values
            scaled_logits = scaled_logits - margin_term
        
        # Compute loss
        losses = -F.logsigmoid(scaled_logits)
        
        # Compute rewards for logging
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        return losses, chosen_rewards, rejected_rewards
    
    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval="train",
    ):
        """
        Compute the MODPO loss and metrics for a batch.
        
        Overrides parent method to incorporate margin values from batch.
        """
        metrics = {}
        
        # Get log probabilities
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(model, batch)
        
        with torch.no_grad():
            if self.ref_model is not None:
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.ref_model, batch
                )
            else:
                # Use model with LoRA disabled as reference
                with self.accelerator.unwrap_model(model).disable_adapter():
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                        model, batch
                    )
        
        # Get margin values if present in batch
        margin_values = batch.get("margin_values", None)
        if margin_values is not None:
            margin_values = margin_values.to(self.accelerator.device)
        
        # Compute MODPO loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            margin_values=margin_values,
        )
        
        # Aggregate loss
        loss = losses.mean()
        
        # Compute metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        
        return loss, metrics
