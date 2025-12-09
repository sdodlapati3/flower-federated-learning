#!/bin/bash
#SBATCH --job-name=fl-pandas
#SBATCH --output=pandas_%j.out
#SBATCH --error=pandas_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/pandas-quickstart

echo "=========================================="
echo "Flower Pandas Quickstart Tutorial"
echo "Federated Analytics with Pandas"
echo "=========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Run Flower Pandas federated analytics simulation
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=3"

echo ""
echo "Pandas simulation completed at: $(date)"
