#!/bin/bash
#SBATCH --job-name=fl-sklearn
#SBATCH --output=sklearn_%j.out
#SBATCH --error=sklearn_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/sklearn-quickstart

echo "=========================================="
echo "Flower scikit-learn Quickstart Tutorial"
echo "Federated Logistic Regression on MNIST"
echo "=========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Run Flower scikit-learn federated learning simulation
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=5"

echo ""
echo "scikit-learn simulation completed at: $(date)"
