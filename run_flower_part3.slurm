#!/bin/bash
#SBATCH --job-name=flower-p3
#SBATCH --output=flower_part3_%j.out
#SBATCH --error=flower_part3_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/flower-tutorial

# Swap server_app for Part 3
cp flower_tutorial/server_app.py flower_tutorial/server_app_backup.py
cp flower_tutorial/server_app_part3.py flower_tutorial/server_app.py

# Run Flower federated learning simulation - Part 3
echo "Starting Flower Federated Learning - Part 3"
echo "Features: Advanced Strategy + W&B Logging + Model Checkpointing"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Set W&B to offline mode (no login required for HPC)
export WANDB_MODE=offline

crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=10"

# Restore original server_app
cp flower_tutorial/server_app_backup.py flower_tutorial/server_app.py

echo ""
echo "Simulation completed at: $(date)"
echo "Check outputs/ directory for model checkpoints"
echo "Check wandb/ directory for offline W&B logs"
