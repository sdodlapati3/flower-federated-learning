#!/bin/bash
#SBATCH --job-name=secagg
#SBATCH --output=secagg_%j.out
#SBATCH --error=secagg_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

echo "=============================================="
echo "Flower Secure Aggregation Example"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/secure-aggregation

echo "SecAgg: Cryptographic privacy for model updates"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
