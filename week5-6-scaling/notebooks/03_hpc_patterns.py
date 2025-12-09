"""
HPC Patterns for Federated Learning
=====================================

Best practices and patterns for running Federated Learning
experiments on High-Performance Computing (HPC) clusters.

Topics covered:
1. SLURM job management for FL experiments
2. Resource allocation strategies
3. Multi-node FL simulation
4. Experiment tracking and logging
5. Handling failures and checkpointing
"""

from typing import Dict, List, Optional
import os
import json
from datetime import datetime


# ============================================================
# Section 1: SLURM Job Templates
# ============================================================

def generate_slurm_script(
    job_name: str,
    python_script: str,
    partition: str = "cpuspot",
    nodes: int = 1,
    cpus: int = 8,
    memory: str = "32G",
    time: str = "04:00:00",
    gpu: bool = False,
    array_size: Optional[int] = None,
    conda_env: str = "~/envs/flower_env"
) -> str:
    """
    Generate a SLURM job script for FL experiments.
    
    Args:
        job_name: Name for the job
        python_script: Path to Python script to run
        partition: SLURM partition
        nodes: Number of nodes
        cpus: CPUs per task
        memory: Memory allocation
        time: Time limit (HH:MM:SS)
        gpu: Whether to request GPU
        array_size: If set, creates job array
        conda_env: Path to conda/crun environment
    
    Returns:
        SLURM script as string
    """
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
"""
    
    if gpu:
        script += "#SBATCH --gres=gpu:1\n"
    
    if array_size:
        script += f"#SBATCH --array=0-{array_size-1}\n"
    
    script += f"""
# ==============================================================
# {job_name}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ==============================================================

echo "=========================================="
echo "Job: {job_name}"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"""
    
    if array_size:
        script += """
# Array job - pass task ID as argument
TASK_ID=$SLURM_ARRAY_TASK_ID
echo "Array Task ID: $TASK_ID"
"""
    
    script += f"""
# Run the experiment
crun -p {conda_env} python {python_script}"""
    
    if array_size:
        script += " --task-id $TASK_ID"
    
    script += """

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
"""
    
    return script


# ============================================================
# Section 2: Experiment Configuration
# ============================================================

class ExperimentConfig:
    """
    Configuration manager for FL experiments.
    
    Handles:
    - Hyperparameter specification
    - Grid search / random search setup
    - Configuration serialization
    """
    
    def __init__(self, name: str, base_config: Dict):
        self.name = name
        self.base_config = base_config
        self.sweep_params = {}
    
    def add_sweep(self, param: str, values: List):
        """Add parameter to sweep over."""
        self.sweep_params[param] = values
    
    def generate_configs(self) -> List[Dict]:
        """Generate all configurations for sweep."""
        from itertools import product
        
        if not self.sweep_params:
            return [self.base_config.copy()]
        
        # Generate all combinations
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        configs = []
        for combo in product(*param_values):
            config = self.base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value
            configs.append(config)
        
        return configs
    
    def save_configs(self, directory: str):
        """Save all configurations to JSON files."""
        configs = self.generate_configs()
        os.makedirs(directory, exist_ok=True)
        
        manifest = []
        for i, config in enumerate(configs):
            filename = f"config_{i:04d}.json"
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            manifest.append({
                'id': i,
                'file': filename,
                'params': {k: config[k] for k in self.sweep_params.keys()}
            })
        
        # Save manifest
        with open(os.path.join(directory, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Generated {len(configs)} configurations in {directory}")
        return configs


# ============================================================
# Section 3: Checkpointing
# ============================================================

class FLCheckpointer:
    """
    Checkpointing for federated learning experiments.
    
    Saves:
    - Global model weights
    - Round number
    - Training metrics
    - Random states for reproducibility
    """
    
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        round_num: int,
        model_weights: Dict,
        metrics: Dict,
        config: Dict
    ):
        """Save a checkpoint."""
        import torch
        
        checkpoint = {
            'round': round_num,
            'model_weights': model_weights,
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.experiment_name}_round_{round_num:04d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
        
        # Also save latest pointer
        latest_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pt")
        torch.save(checkpoint, latest_path)
    
    def load_latest(self) -> Optional[Dict]:
        """Load the latest checkpoint."""
        import torch
        
        latest_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pt")
        
        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path)
            print(f"Loaded checkpoint from round {checkpoint['round']}")
            return checkpoint
        
        return None
    
    def load_checkpoint(self, round_num: int) -> Optional[Dict]:
        """Load a specific checkpoint."""
        import torch
        
        filename = f"{self.experiment_name}_round_{round_num:04d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if os.path.exists(filepath):
            return torch.load(filepath)
        
        return None


# ============================================================
# Section 4: Logging and Metrics
# ============================================================

class FLMetricsLogger:
    """
    Structured logging for FL experiments.
    
    Logs to:
    - JSON file for analysis
    - Console for monitoring
    - Optional W&B integration
    """
    
    def __init__(self, log_dir: str, experiment_name: str, use_wandb: bool = False):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{experiment_name}.jsonl")
        self.metrics_history = []
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("W&B not available, falling back to file logging")
                self.use_wandb = False
    
    def log(self, round_num: int, metrics: Dict, phase: str = "training"):
        """Log metrics for a round."""
        entry = {
            'round': round_num,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(entry)
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=round_num)
        
        # Console output
        metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics.items())
        print(f"[Round {round_num}] {phase}: {metric_str}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}
        
        summary = {
            'total_rounds': len(self.metrics_history),
            'final_accuracy': None,
            'best_accuracy': None,
            'best_round': None
        }
        
        accuracies = [(m['round'], m.get('accuracy', 0)) 
                      for m in self.metrics_history if 'accuracy' in m]
        
        if accuracies:
            summary['final_accuracy'] = accuracies[-1][1]
            best_round, best_acc = max(accuracies, key=lambda x: x[1])
            summary['best_accuracy'] = best_acc
            summary['best_round'] = best_round
        
        return summary


# ============================================================
# Section 5: Resource Monitoring
# ============================================================

def get_resource_usage() -> Dict:
    """Get current resource usage."""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    usage = {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'memory_percent': memory.percent
    }
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            usage['gpu_available'] = True
            usage['gpu_count'] = torch.cuda.device_count()
            usage['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
    except:
        usage['gpu_available'] = False
    
    return usage


# ============================================================
# Section 6: Pattern Examples
# ============================================================

def example_hyperparameter_sweep():
    """Example: Setting up a hyperparameter sweep."""
    
    print("=" * 70)
    print("EXAMPLE: Hyperparameter Sweep Setup")
    print("=" * 70)
    
    # Base configuration
    base_config = {
        'algorithm': 'fedprox',
        'num_rounds': 50,
        'num_clients': 10,
        'clients_per_round': 5,
        'local_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.01,
        'mu': 0.01,
        'alpha': 0.5,
        'seed': 42
    }
    
    # Create sweep
    exp = ExperimentConfig("fedprox_sweep", base_config)
    exp.add_sweep('mu', [0.001, 0.01, 0.1])
    exp.add_sweep('alpha', [0.1, 0.5, 1.0])
    exp.add_sweep('local_epochs', [1, 5, 10])
    
    # Generate configs
    configs = exp.generate_configs()
    print(f"\nGenerated {len(configs)} configurations")
    print(f"Parameters swept: mu, alpha, local_epochs")
    
    # Show sample
    print("\nSample configurations:")
    for i, config in enumerate(configs[:3]):
        print(f"  Config {i}: mu={config['mu']}, alpha={config['alpha']}, epochs={config['local_epochs']}")
    
    return configs


def example_array_job():
    """Example: SLURM array job for parallel experiments."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE: SLURM Array Job")
    print("=" * 70)
    
    # Generate array job script
    script = generate_slurm_script(
        job_name="fl_sweep",
        python_script="run_experiment.py",
        partition="cpuspot",
        cpus=8,
        memory="32G",
        time="04:00:00",
        array_size=27  # For 3x3x3 sweep
    )
    
    print("\nGenerated SLURM array job script:")
    print("-" * 50)
    print(script)
    
    return script


def example_checkpointing():
    """Example: Using checkpointing."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Checkpointing Pattern")
    print("=" * 70)
    
    print("""
Checkpointing workflow:

1. At experiment start:
   ```python
   checkpointer = FLCheckpointer('./checkpoints', 'my_experiment')
   
   # Try to resume
   checkpoint = checkpointer.load_latest()
   if checkpoint:
       start_round = checkpoint['round'] + 1
       model.load_state_dict(checkpoint['model_weights'])
   else:
       start_round = 1
   ```

2. After each round:
   ```python
   for round_num in range(start_round, num_rounds + 1):
       # ... training ...
       
       # Save checkpoint every N rounds
       if round_num % 10 == 0:
           checkpointer.save_checkpoint(
               round_num,
               model.state_dict(),
               {'accuracy': accuracy, 'loss': loss},
               config
           )
   ```

3. On SLURM timeout (signal handling):
   ```python
   import signal
   
   def handle_timeout(signum, frame):
       print("Timeout signal received, saving checkpoint...")
       checkpointer.save_checkpoint(current_round, ...)
       sys.exit(0)
   
   signal.signal(signal.SIGUSR1, handle_timeout)
   ```
""")


def example_multi_node():
    """Example: Multi-node FL simulation."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Multi-Node FL Simulation")
    print("=" * 70)
    
    print("""
Multi-node patterns for FL:

Pattern 1: Server + Client Nodes
--------------------------------
Use separate SLURM jobs for server and clients:

Server job:
```bash
#SBATCH --nodes=1
#SBATCH --job-name=fl_server
crun -p ~/envs/flower python server.py --address 0.0.0.0:8080
```

Client jobs:
```bash
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --job-name=fl_client
SERVER_IP=$(scontrol show job $SERVER_JOB_ID | grep NodeList | ...)
crun -p ~/envs/flower python client.py --server $SERVER_IP:8080 --id $SLURM_ARRAY_TASK_ID
```


Pattern 2: MPI-based FL (simulation)
------------------------------------
```bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
srun crun -p ~/envs/flower python mpi_fl.py
```

In Python:
```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Server logic
    ...
else:
    # Client logic
    ...
```


Pattern 3: Ray-based FL
-----------------------
```python
import ray
ray.init(address="auto")  # Connects to Ray cluster

@ray.remote
def train_client(client_id, weights):
    # Training logic
    return updated_weights

# Parallel client training
futures = [train_client.remote(i, weights) for i in range(10)]
results = ray.get(futures)
```
""")


def main():
    """Run HPC patterns examples."""
    
    print("=" * 70)
    print("HPC PATTERNS FOR FEDERATED LEARNING")
    print("=" * 70)
    
    # Example 1: Hyperparameter sweep
    example_hyperparameter_sweep()
    
    # Example 2: Array job
    example_array_job()
    
    # Example 3: Checkpointing
    example_checkpointing()
    
    # Example 4: Multi-node
    example_multi_node()
    
    # Summary
    print("\n" + "=" * 70)
    print("HPC BEST PRACTICES SUMMARY")
    print("=" * 70)
    print("""
1. JOB MANAGEMENT
   - Use job arrays for parameter sweeps
   - Request appropriate resources (don't over-request)
   - Set time limits based on estimated runtime + buffer
   
2. EXPERIMENT TRACKING
   - Use structured logging (JSON/JSONL)
   - Log hyperparameters with results
   - Use unique experiment IDs
   
3. CHECKPOINTING
   - Save checkpoints regularly (every 10-20 rounds)
   - Include random state for reproducibility
   - Handle SLURM timeout signals
   
4. RESOURCE EFFICIENCY
   - Use CPU-only nodes for simulation
   - Reserve GPUs only for actual training
   - Use array jobs instead of many single jobs
   
5. DEBUGGING
   - Start with short test runs
   - Check output/error files immediately
   - Use interactive sessions for debugging
   
6. REPRODUCIBILITY
   - Fix all random seeds
   - Log software versions
   - Save complete configurations
""")


if __name__ == "__main__":
    main()
