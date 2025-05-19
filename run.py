# run.py
import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Run Absolute Zero with Atropos")
    parser.add_argument("--mode", type=str, choices=["environment", "training", "all"],
                        default="all", help="Which component to run")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model to use for training")
    parser.add_argument("--api-port", type=int, default=8000,
                        help="Port for the Atropos API server")
    parser.add_argument("--env-port", type=int, default=8001,
                        help="Port for the environment server")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of training steps")
    args = parser.parse_args()
    
    processes = []
    
    try:
        # Start API server if running environment or all
        if args.mode in ["environment", "all"]:
            print("Starting Atropos API server...")
            api_proc = subprocess.Popen(
                ["python", "-m", "atropos.api.run_api",
                 "--host", "0.0.0.0", "--port", str(args.api_port)]
            )
            processes.append(api_proc)
            time.sleep(2)  # Give the API server time to start
        
        # Start environment if running environment or all
        if args.mode in ["environment", "all"]:
            print("Starting Absolute Zero environment...")
            env_proc = subprocess.Popen(
                ["python", "scripts/run_environment.py",
                 "--api-port", str(args.api_port),
                 "--env-port", str(args.env_port)]
            )
            processes.append(env_proc)
            time.sleep(2)  # Give the environment time to start
        
        # Start training if running training or all
        if args.mode in ["training", "all"]:
            print("Starting training...")
            train_proc = subprocess.Popen(
                ["python", "scripts/run_training.py",
                 "--model-name", args.model_name,
                 "--batch-size", str(args.batch_size),
                 "--num-steps", str(args.num_steps),
                 "--use-peft", "--use-wandb"]
            )
            processes.append(train_proc)
        
        # Wait for processes to complete
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("Stopping all processes...")
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass
    
    print("All processes stopped")

if __name__ == "__main__":
    main()