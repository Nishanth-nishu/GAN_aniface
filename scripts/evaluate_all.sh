#!/bin/bash
# Script to evaluate all completed GAN models

source /scratch/nishanth.r/gan_proj/venv/bin/activate
cd /scratch/nishanth.r/gan_proj

echo "Starting Evaluation Pipeline"
echo "======================================"

MODELS=("dcgan" "wgan_gp" "sagan" "use_cmhsa")

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python3 gan_anime_faces.py evaluate --model $model
done

echo "======================================"
echo "Evaluation complete! Check outputs/ directory for results."
