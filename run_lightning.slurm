#!/bin/bash
#SBATCH --job-name=fl-lightning
#SBATCH --output=lightning_%j.out
#SBATCH --error=lightning_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/lightning-quickstart

echo "=========================================="
echo "Flower PyTorch Lightning Quickstart Tutorial"
echo "Federated AutoEncoder on MNIST"
echo "=========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Run Flower PyTorch Lightning federated learning simulation
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=3"

echo ""
echo "PyTorch Lightning simulation completed at: $(date)"
