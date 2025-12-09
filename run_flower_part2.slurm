#!/bin/bash
#SBATCH --job-name=flower-p2
#SBATCH --output=flower_part2_%j.out
#SBATCH --error=flower_part2_%j.err
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

# Run Flower federated learning simulation - Part 2
# Using 15 rounds to see learning rate decay in action (every 5 rounds)
echo "Starting Flower Federated Learning - Part 2"
echo "Features: CustomFedAdagrad + Server-side Evaluation + LR Decay"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=15"

echo ""
echo "Simulation completed at: $(date)"
