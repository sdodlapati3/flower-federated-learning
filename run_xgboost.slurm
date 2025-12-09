#!/bin/bash
#SBATCH --job-name=fl-xgboost
#SBATCH --output=xgboost_%j.out
#SBATCH --error=xgboost_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/xgboost-quickstart

echo "=========================================="
echo "Flower XGBoost Quickstart Tutorial"
echo "Federated Gradient Boosting with FedXgbBagging"
echo "=========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Run Flower XGBoost federated learning simulation
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=5"

echo ""
echo "XGBoost simulation completed at: $(date)"
