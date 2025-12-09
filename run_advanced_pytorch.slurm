#!/bin/bash
#SBATCH --job-name=adv-pytorch
#SBATCH --output=advanced_pytorch_%j.out
#SBATCH --error=advanced_pytorch_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/advanced-pytorch

# Run Advanced PyTorch Federated Learning
echo "Starting Advanced PyTorch Federated Learning"
echo "Features:"
echo "  - Non-IID data (Dirichlet partitioning)"
echo "  - Stateful clients (personalized classification head)"
echo "  - Model checkpointing (best accuracy)"
echo "  - W&B logging (offline mode)"
echo "  - Learning rate scheduling"
echo ""
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Set W&B to offline mode
export WANDB_MODE=offline

# Run Flower simulation with 20 clients, 10 rounds
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=10"

echo ""
echo "Simulation completed at: $(date)"
echo "Check outputs/ directory for:"
echo "  - results.json (metrics per round)"
echo "  - run_config.json (configuration)"
echo "  - model_state_*.pth (checkpoints)"
echo "  - final_model.pt"
