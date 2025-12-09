#!/bin/bash
#SBATCH --job-name=flower-p4
#SBATCH --output=flower_part4_%j.out
#SBATCH --error=flower_part4_%j.err
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

# Backup original files
cp flower_tutorial/server_app.py flower_tutorial/server_app_backup.py
cp flower_tutorial/client_app.py flower_tutorial/client_app_backup.py

# Swap to Part 4 implementations
cp flower_tutorial/server_app_part4.py flower_tutorial/server_app.py
cp flower_tutorial/client_app_part4.py flower_tutorial/client_app.py

# Run Flower federated learning simulation - Part 4
echo "Starting Flower Federated Learning - Part 4"
echo "Features: Custom Messages + TrainProcessMetadata + W&B Logging"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Set W&B to offline mode (no login required for HPC)
export WANDB_MODE=offline

crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=10"

# Restore original files
cp flower_tutorial/server_app_backup.py flower_tutorial/server_app.py
cp flower_tutorial/client_app_backup.py flower_tutorial/client_app.py

echo ""
echo "Simulation completed at: $(date)"
echo "Check outputs/ directory for model checkpoints"
echo "Check wandb/ directory for offline W&B logs"
