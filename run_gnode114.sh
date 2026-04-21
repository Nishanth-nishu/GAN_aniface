#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J ANIME_GAN
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode114
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=anime_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

echo "=========================================="
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "START TIME     = $(date)"
echo "=========================================="

# --------------------------------------------------
# Move to project directory
# --------------------------------------------------
cd /scratch/nishanth.r/gan_proj || exit 1
echo "Working directory: $(pwd)"

# --------------------------------------------------
# Activate environment
# --------------------------------------------------

source venv/bin/activate
echo "Activated Conda environment:"
which python
python --version

# Sanity check
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# --------------------------------------------------
# Run full training and evaluation pipeline
# --------------------------------------------------
echo "Starting comprehensive GAN training..."
bash scripts/train_all.sh

echo "Starting evaluation..."
bash scripts/evaluate_all.sh

echo "=========================================="
echo "JOB COMPLETED"
echo "END TIME = $(date)"
echo "=========================================="
