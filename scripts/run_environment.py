# scripts/run_environment.py (continued)
import asyncio
import argparse
from atroposlib.api.run_api import start_server

from atropos_env.config import AbsoluteZeroConfig
from atropos_env.absolute_zero_environment import AbsoluteZeroEnvironment

async def main():
    parser = argparse.ArgumentParser(description="Run the Absolute Zero environment")
    parser.add_argument("--api-host", type=str, default="localhost", 
                        help="Host for the Atropos API server")
    parser.add_argument("--api-port", type=int, default=8000, 
                        help="Port for the Atropos API server")
    parser.add_argument("--env-port", type=int, default=8001, 
                        help="Port for the environment server")
    args = parser.parse_args()
    
    # Create environment config
    config = AbsoluteZeroConfig(
        task_types=["deduction", "abduction", "induction"],
        examples_per_task=3,
        code_executor_timeout=5,
        diversity_weight=0.3,
        difficulty_weight=0.7,
        max_task_buffer_size=1000,
        proposer_probability=0.5,
        # Add other config parameters as needed
    )
    
    # Create environment
    env = AbsoluteZeroEnvironment(config)
    
    # Start environment server
    await env.serve(
        environment_id="absolute_zero",
        api_server=f"http://{args.api_host}:{args.api_port}",
        host="0.0.0.0",
        port=args.env_port
    )

if __name__ == "__main__":
    # Start API server in a separate process
    import multiprocessing
    api_proc = multiprocessing.Process(
        target=start_server,
        kwargs={"host": "0.0.0.0", "port": 8000}
    )
    api_proc.start()
    
    # Run environment
    asyncio.run(main())