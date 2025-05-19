# atropos_env/absolute_zero_environment.py
import random
import json
from typing import Dict, List, Optional, Union, Any


from atropos.environments.base import (
    BaseEnv,
    Item, 
    ScoredDataGroup,
)

from ..data_constructor.task_generator import TaskGenerator
from ..data_constructor.task_validator import TaskValidator
from ..data_constructor.code_executor import CodeExecutor
from ..reward.proposer_reward import ProposerReward
from ..reward.solver_reward import SolverReward
from .config import AbsoluteZeroConfig

class AbsoluteZeroEnvironment(BaseEnv):
    """
    Atropos environment implementing the Absolute Zero paradigm.
    
    This environment switches between two roles:
    1. Task proposer: Generate new reasoning tasks
    2. Task solver: Solve proposed tasks
    
    Both roles are evaluated with appropriate reward functions.
    """
    
    def __init__(self, config: AbsoluteZeroConfig):
        super().__init__(config)
        
        # Initialize components
        self.executor = CodeExecutor(timeout=config.code_executor_timeout)
        self.task_generator = TaskGenerator(task_types=config.task_types)
        self.task_validator = TaskValidator(self.executor)
        self.proposer_reward = ProposerReward(
            diversity_weight=config.diversity_weight,
            difficulty_weight=config.difficulty_weight
        )
        self.solver_reward = SolverReward()
        
        # Task buffers
        self.task_buffer = []
        self.max_buffer_size = config.max_task_buffer_size
        
        # State tracking
        self.current_mode = "proposer"  # Start in proposer mode
        self.current_task = None
        self.current_task_type = None
        self.solver_success_rates = {
            task_type: [] for task_type in config.task_types
        }
    
    async def generate(self, model_provider, **kwargs):
        """Generate prompt based on current mode (proposer/solver)."""
        if self.current_mode == "proposer":
            # Select a task type
            self.current_task_type = random.choice(self.config.task_types)
            
            # Create a task proposal prompt
            prompt = self.task_generator.create_proposal_prompt(
                self.current_task_type, 
                self.config.examples_per_task
            )
            
        else:  # solver mode
            # Get a task from the buffer
            if not self.task_buffer:
                # If buffer is empty, switch to proposer mode
                self.current_mode = "proposer"
                return await self.generate(model_provider, **kwargs)
            
            # Pop a task from the buffer
            self.current_task = self.task_buffer.pop(0)
            self.current_task_type = self.current_task["task_type"]
            
            # Create a solving prompt
            if self.current_task_type == "deduction":
                prompt = self._create_deduction_solving_prompt(self.current_task)
            elif self.current_task_type == "abduction":
                prompt = self._create_abduction_solving_prompt(self.current_task)
            elif self.current_task_type == "induction":
                prompt = self._create_induction_solving_prompt(self.current_task)
            else:
                # Invalid task type, switch back to proposer
                self.current_mode = "proposer"
                return await self.generate(model_provider, **kwargs)
        
        # Generate using the model provider
        return await model_provider.generate(prompt)
    
    async def parse(self, completion, **kwargs):
        """Parse model output based on current mode."""
        if self.current_mode == "proposer":
            # Parse the task proposal
            parsed_task = self.task_generator.parse_proposal(
                self.current_task_type, 
                completion
            )
            
            # Validate the task
            if parsed_task and self.task_validator.validate_task(
                self.current_task_type, 
                parsed_task
            ):
                # Add task type to the parsed task
                parsed_task["task_type"] = self.current_task_type
                
                # Add to task buffer if valid
                if len(self.task_buffer) < self.max_buffer_size:
                    self.task_buffer.append(parsed_task)
                
                # Add to task history
                self.task_generator.add_to_history(
                    self.current_task_type, 
                    parsed_task
                )
                
            return parsed_task
            
        else:  # solver mode
            # Parse the solution
            parsed_solution = self._parse_solution(
                self.current_task_type,
                completion
            )
            
            return {
                "task": self.current_task,
                "solution": parsed_solution
            }
    
    async def score(
        self, parsed_completion, **kwargs
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score based on current mode (proposer or solver)."""
        scores = ScoredDataGroup()
        scores["tokens"] = []  # To be filled based on tokenization
        scores["masks"] = []   # To be filled based on tokenization
        scores["scores"] = []
        
        if self.current_mode == "proposer":
            # For proposer, we need to estimate how challenging this task is
            # by checking historical solver success rates
            if self.current_task_type in self.solver_success_rates:
                success_rates = self.solver_success_rates[self.current_task_type]
                avg_success_rate = sum(success_rates) / max(1, len(success_rates))
            else:
                avg_success_rate = 0.5  # Default if no history
            
            # Compute proposer reward
            reward = self.proposer_reward.compute_reward(
                self.current_task_type,
                parsed_completion,
                avg_success_rate
            )
            
            # Switch to solver mode
            self.current_mode = "solver"
            
        else:  # solver mode
            task = parsed_completion["task"]
            solution = parsed_completion["solution"]
            
            # Execute the solution
            try:
                execution_result = self._execute_solution(
                    self.current_task_type,
                    task,
                    solution
                )
                
                # Compute solver reward
                reward = self.solver_reward.compute_reward(
                    self.current_task_type,
                    task,
                    solution,
                    execution_result
                )
                
                # Update solver success rates
                self.solver_success_rates[self.current_task_type].append(
                    1.0 if reward > 0.5 else 0.0
                )
                # Keep only the most recent N success rates
                max_history = 100
                if len(self.solver_success_rates[self.current_task_type]) > max_history:
                    self.solver_success_rates[self.current_task_type] = \
                        self.solver_success_rates[self.current_task_type][-max_history:]
                
            except Exception as e:
                # Execution failed
                print(f"Solution execution error: {e}")
                reward = 0.0
            
            # Switch to proposer mode with some probability
            if random.random() < self.config.proposer_probability:
                self.current_mode = "proposer"
        
        scores["scores"].append(reward)
        return scores
    
    # Helper methods
    def _create_deduction_solving_prompt(self, task):
        """Create a prompt for solving a deduction task."""
        program = task.get("program", "")
        inputs = task.get("inputs", "")
        return (
            "You are given a Python program and inputs. Your task is to determine "
            "the output of the program when run with these inputs.\n\n"
            f"Program:\n```python\n{program}\n```\n\n"
            f"Inputs: {json.dumps(inputs)}\n\n"
            "What is the output of this program? Explain your reasoning step by step, "
            "then provide the final output value."
        )
    
    def _create_abduction_solving_prompt(self, task):
        """Create a prompt for solving an abduction task."""
        program = task.get("program", "")
        output = task.get("output", "")
        return (
            "You are given a Python program and its output. Your task is to determine "
            "what input would produce this output.\n\n"
            f"Program:\n```python\n{program}\n```\n\n"
            f"Output: {json.dumps(output)}\n\n"
            "What input would produce this output? Explain your reasoning step by step, "
            "then provide the input value."
        )
    
    def _create_induction_solving_prompt(self, task):
        """Create a prompt for solving an induction task."""
        io_pairs = task.get("io_pairs", [])
        io_pairs_str = "\n".join([
            f"Input: {json.dumps(i)}, Output: {json.dumps(o)}"
            for i, o in io_pairs
        ])
        return (
            "You are given several input-output pairs. Your task is to write a Python "
            "program that produces the given output for each corresponding input.\n\n"
            f"Input-Output Pairs:\n{io_pairs_str}\n\n"
            "Write a Python program that satisfies these input-output relationships. "
            "Explain your approach, then provide the final program."
        )
    
    def _parse_solution(self, task_type, completion):
        """Parse the solution based on task type."""
        # Extract solution from the completion
        # This will depend on your model's output format
        if task_type == "deduction":
            # Extract predicted output
            # Simple implementation - can be more sophisticated
            output_match = re.search(r"Final Output:(.+?)$", completion, re.MULTILINE)
            if output_match:
                return output_match.group(1).strip()
            return completion  # Fall back to full completion
            
        elif task_type == "abduction":
            # Extract predicted input
            input_match = re.search(r"Input:(.+?)$", completion, re.MULTILINE)
            if input_match:
                return input_match.group(1).strip()
            return completion
            
        elif task_type == "induction":
            # Extract the program
            code_blocks = re.findall(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
            if code_blocks:
                return code_blocks[0].strip()
            return completion
        
        return completion
    
    def _execute_solution(self, task_type, task, solution):
        """Execute the solution based on task type."""
        if task_type == "deduction":
            # For deduction, solution is the predicted output
            # We don't need to execute anything
            return solution
            
        elif task_type == "abduction":
            # For abduction, solution is the predicted input
            # We need to execute the program with this input
            program = task.get("program", "")
            try:
                # Parse the input from the solution
                input_value = json.loads(solution)
            except:
                # If parsing fails, use the solution as is
                input_value = solution
                
            # Execute the program with the input
            return self.executor.execute(program, input_value)
            
        elif task_type == "induction":
            # For induction, solution is a program
            # We need to execute it with all I/O pairs
            io_pairs = task.get("io_pairs", [])
            
            # Execute the program with each input
            results = {}
            for input_val, _ in io_pairs:
                result = self.executor.execute(solution, input_val)
                results[input_val] = result
                
            return results
            
        return None