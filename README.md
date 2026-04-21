# GAN_aniface

## Overview
This project implements a rigorous, end-to-end framework for generating anime faces using state-of-the-art Generative Adversarial Networks (GANs). Built with PyTorch and adopting robust **SOLID** design principles, the codebase allows for easy experimentation, evaluation, and direct comparison of different generative architectures.

### Architectures Implemented
The platform benchmarks four distinct architectures against the Kaggle Anime Face dataset:
1. **DCGAN**: The foundational convolutional baseline.
2. **WGAN-GP**: Improves stability via the Wasserstein distance and a gradient penalty.
3. **SAGAN**: Introduces Self-Attention mechanisms and Spectral Normalization.
4. **USE-CMHSA-GAN** (Research-Backed—2024): Integrates upsampling squeeze-and-excitation alongside multi-head self-attention for superior spatial coherence.

## Codebase Architecture
The codebase was designed to decouple the data, model architecture, training loops, and evaluation logic. 
It uses a combination of data-classes for configurations, abstract base models for network definitions, and the template method pattern inside a unified trainer for clean training procedures.

## Evaluation
The platform uses the **Fréchet Inception Distance (FID)** metric via the `clean-fid` library to accurately compute distances between real and generated distributions automatically.

## Usage
**1. Train a model:**
```bash
python3 gan_anime_faces.py train --model sagan
```

**2. Evaluate models:**
```bash
python3 gan_anime_faces.py evaluate --all-models
```

**3. Run Slurm Batch:**
If you are running in an HPC environment:
```bash
sbatch run_gnode114.sh
```
