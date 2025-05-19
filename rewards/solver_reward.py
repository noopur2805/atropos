# reward/solver_reward.py
from typing import Dict, Any

class SolverReward:
    """
    Reward function for task solvers, focusing on correctness.
    
    For solver rewards, we primarily care about whether the solution correctly
    solves the proposed task, which is verified through execution.
    """
    
    def __init__(self):
        pass
    
    def compute_reward(self, 
                       task_type: str, 
                       task: Dict[str, Any], 
                       solution: str, 
                       execution_result: Any) -> float:
        """
        Compute reward for a task solution.
        
        Args:
            task_type: Type of reasoning task
            task: Original task dictionary
            solution: Model's solution
            execution_result: Result of executing the solution
            
        Returns:
            Binary reward (0.0 or 1.0) for incorrect/correct
        """
        if task_type == "deduction":
            return self._evaluate_deduction(task, solution, execution_result)
        elif task_type == "abduction":
            return self._evaluate_abduction(task, solution, execution_result)
        elif task_type == "induction":
            return self._evaluate_induction(task, solution, execution_result)
        else:
            return 0.0
    
    def _evaluate_deduction(self, 
                           task: Dict[str, Any], 
                           solution: str,
                           execution_result: Any) -> float:
        """
        Evaluate correctness of deduction task solution.
        
        For deduction: Check if predicted output matches actual output
        """
        expected_output = task.get("output")
        return 1.0 if execution_result == expected_output else 0.0
    
    def _evaluate_abduction(self, 
                           task: Dict[str, Any], 
                           solution: str,
                           execution_result: Any) -> float:
        """
        Evaluate correctness of abduction task solution.
        
        For abduction: Check if reverse-engineered input produces correct output
        """
        program = task.get("program", "")
        expected_output = task.get("output")
        
        # Solution should be an input that produces the expected output
        # execution_result should be the output when program is run with solution
        return 1.0 if execution_result == expected_output else 0.0
    
    def _evaluate_induction(self, 
                           task: Dict[str, Any], 
                           solution: str,
                           execution_result: Any) -> float:
        """
        Evaluate correctness of induction task solution.
        
        For induction: Check if synthesized program correctly handles all I/O pairs
        """
        io_pairs = task.get("io_pairs", [])
        
        # Check if the solution (a program) correctly handles all I/O pairs
        correct_count = 0
        for input_val, expected_output in io_pairs:
            if execution_result.get(input_val) == expected_output:
                correct_count += 1
        
        # All pairs must be correct for full reward
        return 1.0 if correct_count == len(io_pairs) else 0.0