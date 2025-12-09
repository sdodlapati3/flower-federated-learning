#!/bin/bash
#SBATCH --job-name=dp-experiments
#SBATCH --output=dp_experiments_%j.out
#SBATCH --error=dp_experiments_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=============================================="
echo "Differential Privacy Experiments"
echo "IID vs Non-IID with Multiple Epsilon Settings"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

cd /home/sdodl001_odu_edu/flower-federated-learning/dp-experiments

# Create results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Pre-download CIFAR-10
echo "Pre-downloading CIFAR-10 dataset..."
mkdir -p data
crun -p ~/envs/flower_env python -c "
from torchvision import datasets
datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False, download=True)
print('Dataset ready!')
" 2>&1 | tail -3

echo ""
echo "=========================================="
echo "EXPERIMENT MATRIX"
echo "=========================================="
echo ""
echo "IID Experiments (uniform data distribution):"
echo "  - eps2_iid:  ε ≈ 2  (strong privacy, noise_mult=2.0)"
echo "  - eps4_iid:  ε ≈ 4  (moderate privacy, noise_mult=1.0)"
echo "  - eps8_iid:  ε ≈ 8  (weak privacy, noise_mult=0.5)"
echo "  - eps50_iid: ε ≈ 50 (minimal privacy, noise_mult=0.1)"
echo ""
echo "Non-IID Experiments (Dirichlet α=0.5):"
echo "  - eps2_noniid:  ε ≈ 2"
echo "  - eps4_noniid:  ε ≈ 4"
echo "  - eps8_noniid:  ε ≈ 8"
echo "  - eps50_noniid: ε ≈ 50"
echo ""
echo "=========================================="

# Function to run experiment
run_experiment() {
    local name=$1
    local noise_mult=$2
    local is_iid=$3
    local alpha=$4
    
    echo ""
    echo ">>> Running: $name (noise=$noise_mult, iid=$is_iid, alpha=$alpha)"
    echo "-------------------------------------------"
    
    crun -p ~/envs/flower_env -- flwr run . \
        --run-config "experiment-name=$name noise-multiplier=$noise_mult is-iid=$is_iid dirichlet-alpha=$alpha" \
        2>&1 | tee "$RESULTS_DIR/${name}.log"
    
    echo ">>> Completed: $name"
    echo ""
}

# Run IID experiments
echo ""
echo "========== IID EXPERIMENTS =========="
run_experiment "iid_eps50" 0.1 true 1.0
run_experiment "iid_eps8" 0.5 true 1.0
run_experiment "iid_eps4" 1.0 true 1.0
run_experiment "iid_eps2" 2.0 true 1.0

# Run Non-IID experiments
echo ""
echo "========== NON-IID EXPERIMENTS =========="
run_experiment "noniid_eps50" 0.1 false 0.5
run_experiment "noniid_eps8" 0.5 false 0.5
run_experiment "noniid_eps4" 1.0 false 0.5
run_experiment "noniid_eps2" 2.0 false 0.5

# Generate summary
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="

echo ""
echo "Extracting final accuracies..."
echo ""

for log in $RESULTS_DIR/*.log; do
    exp_name=$(basename $log .log)
    accuracy=$(grep -oP "accuracy.*?(\d+\.\d+)" $log | tail -1 | grep -oP "\d+\.\d+")
    epsilon=$(grep -oP "ε = \d+\.\d+" $log | tail -1 | grep -oP "\d+\.\d+")
    echo "$exp_name: accuracy=$accuracy, ε=$epsilon"
done | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "=========================================="
echo "EXPECTED FINDINGS"
echo "=========================================="
echo "1. Lower ε → stronger privacy → lower accuracy"
echo "2. Non-IID → harder learning → lower accuracy than IID"
echo "3. Privacy-utility tradeoff is more severe for Non-IID"
echo ""

echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "Job completed!"
