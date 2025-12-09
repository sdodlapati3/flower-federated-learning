#!/bin/bash
#SBATCH --job-name=dp-exp-standalone
#SBATCH --output=dp_standalone_%j.out
#SBATCH --error=dp_standalone_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=============================================="
echo "DP Experiments - Standalone Python"
echo "Multiple Epsilon × IID vs Non-IID"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/dp-experiments

# Run the standalone Python experiment
crun -p ~/envs/flower_env python run_standalone_experiments.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
