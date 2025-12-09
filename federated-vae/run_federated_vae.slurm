#!/bin/bash
#SBATCH --job-name=fed-vae
#SBATCH --output=fed_vae_%j.out
#SBATCH --error=fed_vae_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=============================================="
echo "Federated Variational Autoencoder (VAE)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/federated-vae

echo "VAE: Unsupervised learning for image reconstruction"
echo "Configuration:"
echo "  - Clients: 2"
echo "  - Rounds: 5"
echo "  - Dataset: CIFAR-10"
echo ""

crun -p ~/envs/flower_env -- flwr run .

echo ""
echo "End time: $(date)"
echo "Job completed!"
