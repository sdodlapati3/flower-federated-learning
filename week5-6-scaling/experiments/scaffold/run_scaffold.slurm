#!/bin/bash
#SBATCH --job-name=scaffold_exp
#SBATCH --output=scaffold_%j.out
#SBATCH --error=scaffold_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==============================================================
# SCAFFOLD Experiment
# Comparing FedAvg vs SCAFFOLD for variance reduction
# ==============================================================

echo "=========================================="
echo "SCAFFOLD Experiment"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "=========================================="

# Navigate to experiment directory
cd /home/sdodl001_odu_edu/flower-federated-learning/week5-6-scaling/experiments/scaffold

# Run the experiment using crun
crun -p ~/envs/flower_env python scaffold_implementation.py

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
