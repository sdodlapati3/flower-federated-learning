#!/bin/bash
#SBATCH --job-name=vertical-fl
#SBATCH --output=vertical_fl_%j.out
#SBATCH --error=vertical_fl_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

echo "=============================================="
echo "Flower Vertical Federated Learning (Split Learning)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/vertical-fl

echo "Vertical FL: Each client has DIFFERENT FEATURES for the SAME samples"
echo "Configuration:"
echo "  - Clients: 3 (each with different feature subsets)"
echo "  - Feature splits: [2, 3, 2] (total 7 features)"
echo "  - Rounds: 100"
echo "  - Dataset: Titanic"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
