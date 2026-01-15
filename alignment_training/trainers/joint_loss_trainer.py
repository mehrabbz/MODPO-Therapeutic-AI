"""
Joint-Loss DPO Trainer

A multi-objective DPO variant that combines multiple preference objectives into
a single sigmoid-weighted loss function, treating all criteria as primary objectives
rather than using the margin-based approach of MODPO.

This approach trains on multiple preference datasets simultaneously with weighted
contributions to the final loss.
"""

import torch
import torch.nn.functional as F
from trl import DPOTrainer


class JointLossDPOTrainer(DPOTrainer):
    """
    Joint-Loss DPO Trainer for multi-objective preference optimization.
    
    Unlike MODPO which uses margin rewards from auxiliary models, Joint-Loss DPO
    directly combines multiple DPO losses with learnable or fixed weights:
    
    L_joint = -log σ(Σ_k β * w_k * (log π(y_w^k|x)/π_ref(y_w^k|x) - log π(y_l^k|x)/π_ref(y_l^k|x)))
    
    where each objective k has its own chosen/rejected pairs and weight w_k.
    
    Args:
        model: The model to train
        ref_model: Reference model (optional)
        weights: List of weights for each objective
        num_objectives: Number of objectives being optimized
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
        
        self.num_objectives = num_objectives
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
        
        print(f"Joint-Loss DPO initialized with {num_objectives} objectives")
        print(f"Weights: {self.weights.tolist()}, Beta: {self.beta}")
    
    def joint_dpo_loss(
        self,
        all_policy_chosen_logps: list[torch.FloatTensor],
        all_policy_rejected_logps: list[torch.FloatTensor],
        all_reference_chosen_logps: list[torch.FloatTensor],
        all_reference_rejected_logps: list[torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the joint loss across all objectives.
        
        Args:
            all_policy_chosen_logps: List of chosen log probs for each objective
            all_policy_rejected_logps: List of rejected log probs for each objective
            all_reference_chosen_logps: List of reference chosen log probs
            all_reference_rejected_logps: List of reference rejected log probs
            
        Returns:
            losses: Per-example losses
            chosen_rewards: Average rewards for chosen responses
            rejected_rewards: Average rewards for rejected responses
        """
        # Compute weighted sum of logits across all objectives
        combined_logits = torch.zeros_like(all_policy_chosen_logps[0])
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        for k in range(self.num_objectives):
            # Compute log ratios for objective k
            policy_diff = all_policy_chosen_logps[k] - all_policy_rejected_logps[k]
            reference_diff = all_reference_chosen_logps[k] - all_reference_rejected_logps[k]
            
            logits_k = policy_diff - reference_diff
            
            # Add weighted contribution
            combined_logits = combined_logits + self.beta * self.weights[k] * logits_k
            
            # Track rewards for logging
            chosen_rewards_k = self.beta * (all_policy_chosen_logps[k] - all_reference_chosen_logps[k])
            rejected_rewards_k = self.beta * (all_policy_rejected_logps[k] - all_reference_rejected_logps[k])
            
            all_chosen_rewards.append(chosen_rewards_k)
            all_rejected_rewards.append(rejected_rewards_k)
        
        # Compute loss with combined logits
        losses = -F.logsigmoid(combined_logits)
        
        # Average rewards across objectives for logging
        chosen_rewards = torch.stack(all_chosen_rewards).mean(dim=0)
        rejected_rewards = torch.stack(all_rejected_rewards).mean(dim=0)
        
        return losses, chosen_rewards, rejected_rewards
    
    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval="train",
    ):
        """
        Compute the Joint-Loss DPO loss and metrics for a batch.
        
        Expects batch to contain keys for each objective:
        - chosen_input_ids_0, chosen_attention_mask_0, rejected_input_ids_0, ... (objective 0)
        - chosen_input_ids_1, chosen_attention_mask_1, rejected_input_ids_1, ... (objective 1)
        - etc.
        """
        metrics = {}
        
        all_policy_chosen_logps = []
        all_policy_rejected_logps = []
        all_reference_chosen_logps = []
        all_reference_rejected_logps = []
        
        # Process each objective
        for k in range(self.num_objectives):
            # Create sub-batch for objective k
            sub_batch = {
                "chosen_input_ids": batch[f"chosen_input_ids_{k}"],
                "chosen_attention_mask": batch[f"chosen_attention_mask_{k}"],
                "rejected_input_ids": batch[f"rejected_input_ids_{k}"],
                "rejected_attention_mask": batch[f"rejected_attention_mask_{k}"],
            }
            
            # Get policy log probs
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(model, sub_batch)
            all_policy_chosen_logps.append(policy_chosen_logps)
            all_policy_rejected_logps.append(policy_rejected_logps)
            
            # Get reference log probs
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(
                        self.ref_model, sub_batch
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(
                            model, sub_batch
                        )
            
            all_reference_chosen_logps.append(ref_chosen_logps)
            all_reference_rejected_logps.append(ref_rejected_logps)
        
        # Compute joint loss
        losses, chosen_rewards, rejected_rewards = self.joint_dpo_loss(
            all_policy_chosen_logps,
            all_policy_rejected_logps,
            all_reference_chosen_logps,
            all_reference_rejected_logps,
        )
        
        loss = losses.mean()
        
        # Compute metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        
        # Per-objective metrics
        for k in range(self.num_objectives):
            metrics[f"{prefix}logps/chosen_obj{k}"] = all_policy_chosen_logps[k].mean().item()
            metrics[f"{prefix}logps/rejected_obj{k}"] = all_policy_rejected_logps[k].mean().item()
        
        return loss, metrics
