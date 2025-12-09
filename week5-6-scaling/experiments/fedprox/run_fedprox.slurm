#!/bin/bash
#SBATCH --job-name=fedprox_exp
#SBATCH --output=fedprox_%j.out
#SBATCH --error=fedprox_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==============================================================
# FedProx Experiment
# Comparing FedAvg vs FedProx across different heterogeneity levels
# ==============================================================

echo "=========================================="
echo "FedProx Experiment"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "=========================================="

# Navigate to experiment directory
cd /home/sdodl001_odu_edu/flower-federated-learning/week5-6-scaling/experiments/fedprox

# Run the experiment using crun
crun -p ~/envs/flower_env python fedprox_implementation.py

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
