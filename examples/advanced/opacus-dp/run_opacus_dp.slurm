#!/bin/bash
#SBATCH --job-name=opacus-dp
#SBATCH --output=opacus_dp_%j.out
#SBATCH --error=opacus_dp_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=============================================="
echo "Opacus Differential Privacy Federated Learning"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Navigate to example directory
cd /home/sdodl001_odu_edu/flower-federated-learning/opacus-dp

# Run Flower simulation with Opacus DP
echo "Starting FL simulation with Differential Privacy..."
echo "Configuration:"
echo "  - Clients: 2"
echo "  - Rounds: 5"
echo "  - Target Delta: 1e-5"
echo "  - Max Grad Norm: 1.0"
echo ""

crun -p ~/envs/flower_env -- flwr run . --run-config "num-server-rounds=5"

echo ""
echo "End time: $(date)"
echo "Job completed!"
