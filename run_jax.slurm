#!/bin/bash
#SBATCH --job-name=fl-jax
#SBATCH --output=jax_%j.out
#SBATCH --error=jax_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/jax-quickstart

echo "=========================================="
echo "Flower JAX Quickstart Tutorial"
echo "Federated Linear Regression with JAX"
echo "=========================================="
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Run Flower JAX federated learning simulation
crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=5"

echo ""
echo "JAX simulation completed at: $(date)"
