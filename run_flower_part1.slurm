#!/bin/bash
#SBATCH --job-name=flower-p1
#SBATCH --output=flower_part1_%j.out
#SBATCH --error=flower_part1_%j.err
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

# Temporarily swap server_app for Part 1
cp flower_tutorial/server_app.py flower_tutorial/server_app_part2_backup.py
cp flower_tutorial/server_app_part1.py flower_tutorial/server_app.py

# Run Flower federated learning simulation - Part 1
echo "Starting Flower Federated Learning - Part 1"
echo "Features: FedAvg (basic federated averaging)"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

crun -p ~/envs/flower_env flwr run . --run-config="num-server-rounds=3"

# Restore Part 2 server_app
cp flower_tutorial/server_app_part2_backup.py flower_tutorial/server_app.py

echo ""
echo "Simulation completed at: $(date)"
