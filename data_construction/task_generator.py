# data_constructor/task_generator.py
import random
from typing import List, Dict, Any

class TaskGenerator:
    """Handles the generation and parsing of task proposals."""
    
    def __init__(self, task_types=["deduction", "abduction", "induction"]):
        self.task_types = task_types
        self.task_history = {task_type: [] for task_type in task_types}
        
    def create_proposal_prompt(self, task_type: str, num_examples: int = 3) -> str:
        """
        Create a prompt for the model to propose a new task of the given type.
        
        Args:
            task_type: Type of reasoning task to generate
            num_examples: Number of previous examples to include
        
        Returns:
            Complete prompt for task proposal
        """
        # Get examples of this task type to show
        examples = self.get_examples(task_type, num_examples)
        
        # Create appropriate prompt based on task type
        if task_type == "deduction":
            return self._create_deduction_proposal_prompt(examples)
        elif task_type == "abduction":
            return self._create_abduction_proposal_prompt(examples)
        elif task_type == "induction":
            return self._create_induction_proposal_prompt(examples)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def parse_proposal(self, task_type: str, completion: str) -> Dict[str, Any]:
        """
        Parse the model's task proposal into structured components.
        
        Args:
            task_type: Type of reasoning task that was generated
            completion: Raw model completion containing the task proposal
            
        Returns:
            Structured task dictionary
        """
        # Extract task components based on task type
        if task_type == "deduction":
            return self._parse_deduction_proposal(completion)
        elif task_type == "abduction":
            return self._parse_abduction_proposal(completion)
        elif task_type == "induction":
            return self._parse_induction_proposal(completion)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def add_to_history(self, task_type: str, task: Dict[str, Any]) -> None:
        """Add a successfully generated task to the history."""
        self.task_history[task_type].append(task)
        
    def get_examples(self, task_type: str, num_examples: int) -> List[Dict[str, Any]]:
        """Get examples from history for a particular task type."""
        history = self.task_history[task_type]
        if not history:
            return []
        return random.sample(history, min(num_examples, len(history)))
    
    # Private helper methods for each task type
    def _create_deduction_proposal_prompt(self, examples: List[Dict[str, Any]]) -> str:
        # Implementation for deduction task prompt generation
        pass
        
    def _create_abduction_proposal_prompt(self, examples: List[Dict[str, Any]]) -> str:
        # Implementation for abduction task prompt generation
        pass
        
    def _create_induction_proposal_prompt(self, examples: List[Dict[str, Any]]) -> str:
        # Implementation for induction task prompt generation
        pass
    
    def _parse_deduction_proposal(self, completion: str) -> Dict[str, Any]:
        # Implementation for parsing deduction task proposals
        pass
        
    def _parse_abduction_proposal(self, completion: str) -> Dict[str, Any]:
        # Implementation for parsing abduction task proposals
        pass
        
    def _parse_induction_proposal(self, completion: str) -> Dict[str, Any]:
        # Implementation for parsing induction task proposals
        pass