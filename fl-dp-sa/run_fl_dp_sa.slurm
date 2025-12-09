#!/bin/bash
#SBATCH --job-name=fl-dp-sa
#SBATCH --output=fl_dp_sa_%j.out
#SBATCH --error=fl_dp_sa_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=============================================="
echo "FL with Differential Privacy + Secure Aggregation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Navigate to example directory
cd /home/sdodl001_odu_edu/flower-federated-learning/fl-dp-sa

# Run Flower simulation with DP + SecAgg
echo "Starting FL simulation with DP + Secure Aggregation..."
echo "Configuration:"
echo "  - Clients: 10"
echo "  - Sampled per round: 5"
echo "  - Noise Multiplier: 0.2"
echo "  - Clipping Norm: 10"
echo "  - SecAgg+ Shares: 3"
echo "  - Reconstruction Threshold: 2"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
