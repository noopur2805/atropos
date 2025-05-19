# scripts/run_training.py
import os
import asyncio
import argparse
import json
import random
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import wandb

from atropos.api.run_client import register_trainer, get_batch

class GRPO:
    """
    Group Relative Policy Optimization trainer for the Absolute Zero paradigm.
    """
    
    def __init__(
        self,
        model_name: str,
        peft_config: Optional[Dict] = None,
        learning_rate: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = True,
        wandb_project: str = "absolute_zero_atropos",
        wandb_entity: Optional[str] = None,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # Apply LoRA if peft_config is provided
        if peft_config:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **peft_config
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize WandB
        if use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "peft_config": peft_config,
                    "device": device,
                }
            )
    
    async def train(
        self,
        num_steps: int,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        log_interval: int = 10,
        save_interval: int = 100,
        evaluation_interval: int = 100,
    ):
        """
        Train the model using the Atropos API.
        """
        # Register the trainer with the API
        await register_trainer(
            trainer_id=f"grpo_{random.randint(0, 10000)}",
            environments=["absolute_zero"],
            model_name=self.model_name,
            batch_size=batch_size
        )
        
        # Training loop
        step = 0
        while step < num_steps:
            try:
                # Get a batch of data
                batch_data = await get_batch()
                
                if not batch_data or not batch_data.get("items"):
                    print("Empty batch received, waiting...")
                    await asyncio.sleep(1)
                    continue
                
                # Process and train on the batch
                train_metrics = await self._train_on_batch(
                    batch_data,
                    gradient_accumulation_steps=gradient_accumulation_steps
                )
                
                # Log metrics
                if step % log_interval == 0:
                    print(f"Step {step}: {train_metrics}")
                    if self.use_wandb:
                        wandb.log(train_metrics, step=step)
                
                # Save checkpoint
                if step % save_interval == 0 and step > 0:
                    self._save_checkpoint(step)
                
                # Evaluate model
                if step % evaluation_interval == 0 and step > 0:
                    eval_metrics = await self._evaluate_model()
                    print(f"Evaluation at step {step}: {eval_metrics}")
                    if self.use_wandb:
                        wandb.log(eval_metrics, step=step)
                
                step += 1
                
            except Exception as e:
                print(f"Error during training: {e}")
                await asyncio.sleep(1)
        
        # Save final checkpoint
        self._save_checkpoint(num_steps, final=True)
        
        if self.use_wandb:
            wandb.finish()
    
    async def _train_on_batch(
        self,
        batch_data: Dict[str, Any],
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Train on a batch of data from the Atropos API.
        """
        self.model.train()
        
        # Extract items and rewards
        items = batch_data.get("items", [])
        
        total_loss = 0
        batch_metrics = {
            "train/loss": 0.0,
            "train/proposer_reward": 0.0,
            "train/solver_reward": 0.0,
            "train/proposer_samples": 0,
            "train/solver_samples": 0,
        }
        
        # Process each item in the batch
        for i, item in enumerate(items):
            # Extract tokens, attention mask, and reward
            tokens = torch.tensor(item.get("tokens", []), dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(item.get("attention_mask", []), dtype=torch.long).to(self.device)
            reward = item.get("reward", 0.0)
            role = item.get("role", "unknown")  # proposer or solver
            
            # Forward pass
            outputs = self.model(
                input_ids=tokens.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                labels=tokens.unsqueeze(0)
            )
            
            # Scale loss based on reward
            loss = outputs.loss * reward
            
            # Scale by gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            total_loss += loss.item()
            batch_metrics["train/loss"] += loss.item()
            
            # Update role-specific metrics
            if role == "proposer":
                batch_metrics["train/proposer_reward"] += reward
                batch_metrics["train/proposer_samples"] += 1
            elif role == "solver":
                batch_metrics["train/solver_reward"] += reward
                batch_metrics["train/solver_samples"] += 1
            
            # Step optimizer if needed
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(items) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Compute average metrics
        batch_metrics["train/loss"] /= len(items)
        
        if batch_metrics["train/proposer_samples"] > 0:
            batch_metrics["train/proposer_reward"] /= batch_metrics["train/proposer_samples"]
        
        if batch_metrics["train/solver_samples"] > 0:
            batch_metrics["train/solver_reward"] /= batch_metrics["train/solver_samples"]
            
        return batch_metrics
    
    async def _evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the model using the Atropos API.
        """
        # For a simple evaluation, we can get a batch without updating the model
        # and compute the loss
        eval_batch = await get_batch()
        
        if not eval_batch or not eval_batch.get("items"):
            return {"eval/loss": 0.0}
        
        self.model.eval()
        
        total_loss = 0.0
        proposer_reward = 0.0
        solver_reward = 0.0
        proposer_samples = 0
        solver_samples = 0
        
        with torch.no_grad():
            for item in eval_batch.get("items", []):
                # Extract tokens, attention mask, and reward
                tokens = torch.tensor(item.get("tokens", []), dtype=torch.long).to(self.device)
                attention_mask = torch.tensor(item.get("attention_mask", []), dtype=torch.long).to(self.device)
                reward = item.get("reward", 0.0)
                role = item.get("role", "unknown")  # proposer or solver
                
                # Forward pass
                outputs = self.model(
                    input_ids=tokens.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    labels=tokens.unsqueeze(0)
                )
                
                # Update metrics
                total_loss += outputs.loss.item()
                
                # Update role-specific metrics
                if role == "proposer":
                    proposer_reward += reward
                    proposer_samples += 1
                elif role == "solver":
                    solver_reward += reward
                    solver_samples += 1
        
        # Compute average metrics
        eval_metrics = {
            "eval/loss": total_loss / len(eval_batch.get("items", [])),
            "eval/proposer_reward": proposer_reward / max(1, proposer_samples),
            "eval/solver_reward": solver_reward / max(1, solver_samples),
        }
        
        return eval_metrics
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """
        Save a checkpoint of the model.
        """
        save_dir = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
        if final:
            save_dir = os.path.join(self.checkpoint_dir, "final_model")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Checkpoint saved at {save_dir}")


async def main():
    parser = argparse.ArgumentParser(description="Train a model with Absolute Zero")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Hugging Face model name or path")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--use-peft", action="store_true",
                        help="Use PEFT/LoRA for efficient training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="absolute_zero_atropos",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity name")
    args = parser.parse_args()
    
    # Create PEFT config if needed
    peft_config = None
    if args.use_peft:
        peft_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
        }
    
    # Create trainer
    trainer = GRPO(
        model_name=args.model_name,
        peft_config=peft_config,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    
    # Train model
    await trainer.train(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

if __name__ == "__main__":
    asyncio.run(main())