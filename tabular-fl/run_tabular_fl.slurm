#!/bin/bash
#SBATCH --job-name=tabular-fl
#SBATCH --output=tabular_fl_%j.out
#SBATCH --error=tabular_fl_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

echo "=============================================="
echo "Federated Learning on Tabular Data"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/tabular-fl

echo "Tabular FL: Binary classification on synthetic census data"
echo "Configuration:"
echo "  - Clients: 5"
echo "  - Rounds: 10"
echo "  - Features: 14"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
