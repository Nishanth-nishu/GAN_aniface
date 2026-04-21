#!/bin/bash
# Script to train all GAN architectures sequentially

# Fail on error
set -e

# Setup environment
source /scratch/nishanth.r/gan_proj/venv/bin/activate
cd /scratch/nishanth.r/gan_proj

echo "Starting Multi-Model GAN Training Pipeline"
echo "======================================"

# Standard arguments
EPOCHS=200
BATCH_SIZE=64

# 1. DCGAN (Baseline)
echo ""
echo "[1/4] Training DCGAN"
echo "--------------------------------------"
python3 gan_anime_faces.py train --model dcgan --epochs $EPOCHS --batch-size 128

# 2. WGAN-GP
echo ""
echo "[2/4] Training WGAN-GP"
echo "--------------------------------------"
python3 gan_anime_faces.py train --model wgan_gp --epochs $EPOCHS --batch-size $BATCH_SIZE

# 3. SAGAN
echo ""
echo "[3/4] Training SAGAN"
echo "--------------------------------------"
python3 gan_anime_faces.py train --model sagan --epochs $EPOCHS --batch-size $BATCH_SIZE

# 4. USE-CMHSA-GAN (State of the Art)
echo ""
echo "[4/4] Training USE-CMHSA-GAN"
echo "--------------------------------------"
python3 gan_anime_faces.py train --model use_cmhsa --epochs $EPOCHS --batch-size $BATCH_SIZE

echo ""
echo "======================================"
echo "All local training runs completed!"
echo "To evaluate, run: bash scripts/evaluate_all.sh"
