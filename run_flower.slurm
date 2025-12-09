#!/bin/bash
#SBATCH --job-name=flower-fl
#SBATCH --output=flower_%j.out
#SBATCH --error=flower_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Load Python module
module load python3/2025.1-py312

# Navigate to project directory
cd /home/sdodl001_odu_edu/flower-federated-learning/flower-tutorial

# Run Flower federated learning simulation
echo "Starting Flower Federated Learning Simulation..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

crun -p ~/envs/flower_env flwr run .

echo ""
echo "Simulation completed at: $(date)"
