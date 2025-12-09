#!/bin/bash
#SBATCH --job-name=paper-replication
#SBATCH --output=paper_replication_%j.out
#SBATCH --error=paper_replication_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=============================================="
echo "Paper Replication: DP in Federated Learning"
echo "Tayyeh et al., 2024"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/week3-4-dp-theory/experiments/paper_replication

# Create data directory
mkdir -p data

# Run the sweep
crun -p ~/envs/flower_env python run_sweep.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
