# reward/proposer_reward.py
from typing import Dict, Any, List
import numpy as np

class ProposerReward:
    """
    Reward function for task proposers, focusing on learnability.
    
    The key idea is to reward tasks that are:
    1. Neither too easy (100% success) nor too hard (0% success)
    2. Sufficiently diverse from previous tasks
    3. Valid and executable
    """
    
    def __init__(self, diversity_weight: float = 0.3, difficulty_weight: float = 0.7):
        self.diversity_weight = diversity_weight
        self.difficulty_weight = difficulty_weight
        self.task_history = {
            "deduction": [],
            "abduction": [],
            "induction": []
        }
    
    def compute_reward(self, 
                       task_type: str, 
                       task: Dict[str, Any], 
                       solver_success_rate: float) -> float:
        """
        Compute reward for a task proposal.
        
        Args:
            task_type: Type of reasoning task
            task: Task dictionary with components
            solver_success_rate: Success rate of solvers on this task (0.0-1.0)
            
        Returns:
            Reward score (0.0-1.0)
        """
        # Calculate difficulty reward
        difficulty_reward = self._compute_difficulty_reward(solver_success_rate)
        
        # Calculate diversity reward
        diversity_reward = self._compute_diversity_reward(task_type, task)
        
        # Combine rewards
        total_reward = (
            self.difficulty_weight * difficulty_reward +
            self.diversity_weight * diversity_reward
        )
        
        # Add to history for future diversity calculations
        self.task_history[task_type].append(task)
        
        return total_reward
    
    def _compute_difficulty_reward(self, solver_success_rate: float) -> float:
        """
        Compute reward based on difficulty (solver success rate).
        
        The ideal task is neither too easy nor too hard.
        Maximum reward at 50% success rate, minimum at 0% or 100%.
        """
        # Using a bell curve centered at 0.5 (50% success rate)
        # 1 - 4 * (x - 0.5)^2 gives a parabola with max at x=0.5
        return 1.0 - 4.0 * (solver_success_rate - 0.5) ** 2
    
    def _compute_diversity_reward(self, task_type: str, task: Dict[str, Any]) -> float:
        """
        Compute reward based on diversity from previous tasks.
        
        Args:
            task_type: Type of reasoning task
            task: Task dictionary with components
            
        Returns:
            Diversity reward (0.0-1.0), higher for more diverse tasks
        """
        if not self.task_history[task_type]:
            return 1.0  # First task is maximally diverse
        
        # Compute similarity to each previous task
        similarities = [
            self._compute_task_similarity(task_type, task, prev_task)
            for prev_task in self.task_history[task_type][-10:]  # Last 10 tasks
        ]
        
        # Reward is inversely proportional to maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity
    
    def _compute_task_similarity(self, 
                                task_type: str, 
                                task1: Dict[str, Any], 
                                task2: Dict[str, Any]) -> float:
        """
        Compute similarity between two tasks (0.0-1.0).
        
        Implementation depends on task type and task representation.
        """
        # Task-specific similarity calculation based on task components
        if task_type == "deduction":
            # Compare program structure, input complexity, etc.
            pass
        elif task_type == "abduction":
            # Compare program structure, output complexity, etc.
            pass
        elif task_type == "induction":
            # Compare I/O pairs pattern complexity
            pass
        else:
            return 0.0
            
        # Default simple implementation - override with more sophisticated metrics
        return 0.2  # Low default similarity