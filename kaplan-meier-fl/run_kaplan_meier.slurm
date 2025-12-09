#!/bin/bash
#SBATCH --job-name=kaplan-meier
#SBATCH --output=kaplan_meier_%j.out
#SBATCH --error=kaplan_meier_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

echo "=============================================="
echo "Federated Kaplan-Meier Survival Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/kaplan-meier-fl

# Install lifelines if not present
crun -p ~/envs/flower_env pip install lifelines 2>&1 | tail -3

echo "Kaplan-Meier: Federated survival curve estimation"
echo "Configuration:"
echo "  - Clients: 2 (one per treatment group)"
echo "  - Dataset: Waltons survival data"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
