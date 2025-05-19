# atropos_env/config.py
from atroposlib.environments.base import BaseEnvConfig
from typing import List

class AbsoluteZeroConfig(BaseEnvConfig):
    """Configuration for the Absolute Zero environment."""
    
    # Task generation settings
    task_types: List[str] = ["deduction", "abduction", "induction"]
    examples_per_task: int = 3
    
    # Code execution settings
    code_executor_timeout: int = 5
    
    # Reward weights
    diversity_weight: float = 0.3
    difficulty_weight: float = 0.7
    
    # Buffer settings
    max_task_buffer_size: int = 1000
    
    # Environment settings
    proposer_probability: float = 0.5  # Probability of selecting proposer mode