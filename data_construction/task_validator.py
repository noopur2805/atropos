# data_constructor/task_validator.py
from .code_executor import CodeExecutor

class TaskValidator:
    """Validates proposed tasks to ensure they are well-formed and executable."""
    
    def __init__(self, executor: CodeExecutor):
        self.executor = executor
    
    def validate_task(self, task_type: str, task: dict) -> bool:
        """
        Validate a proposed task based on its type.
        
        Args:
            task_type: Type of reasoning task
            task: Task dictionary containing components
            
        Returns:
            True if the task is valid, False otherwise
        """
        try:
            if task_type == "deduction":
                # For deduction, we need to verify the program runs with the input
                program = task.get("program", "")
                inputs = task.get("inputs", [])
                expected_output = task.get("output")
                
                # Verify by running the code
                result = self.executor.execute(program, inputs)
                
                # Check if output matches expected output
                return result == expected_output
                
            elif task_type == "abduction":
                # For abduction, verify program produces the output with some input
                program = task.get("program", "")
                expected_output = task.get("output")
                
                # Try to find an input that produces the output
                # This could be the original input or a verification
                return self.executor.verify_abduction(program, expected_output)
                
            elif task_type == "induction":
                # For induction, verify all input-output pairs are consistent
                io_pairs = task.get("io_pairs", [])
                
                # Check if all pairs are valid
                return self.executor.verify_io_pairs(io_pairs)
                
            else:
                return False
                
        except Exception as e:
            # If any errors occur during validation, the task is invalid
            print(f"Task validation error: {e}")
            return False