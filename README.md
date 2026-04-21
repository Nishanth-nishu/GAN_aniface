# Anime Face Generation using Advanced GANs

## 📌 Project Overview
This project establishes a production-grade, end-to-end framework dedicated to generating high-quality anime faces. Drawing entirely upon S.O.L.I.D object-oriented design principles and constructed natively in PyTorch, the repository enables seamless experimentation, mathematical evaluation, and benchmarking across four prominent Generative Adversarial Network architectures. 

The primary dataset deployed is the renowned **Kaggle Anime Face Dataset** (`splcher/animefacedataset`), comprising roughly ~63,000 diverse profile illustrations.

## 🧠 System Architecture & Implementations

Our system strictly decouples logic into discrete segments: `configs/`, `data/`, `evaluation/`, `models/`, and `trainers/`. The generation capability leverages four progressive models, scaling from classical convolutional baselines to contemporary 2024 literature implementations:

1. **DCGAN (Deep Convolutional GAN)**
   - *Paradigm:* The fundamental foundational baseline acting as a structural sanity check.
   - *Core Mechanism:* Leverages completely strided-convolutions for downsampling and `ConvTranspose2d` for spatial construction. Evaluates directly using a dynamic Logit equilibrium (`BCEWithLogitsLoss`).
2. **WGAN-GP (Wasserstein GAN w/ Gradient Penalty)**
   - *Paradigm:* Tackles DCGAN's infamous mode collapse issues by enforcing a 1-Lipschitz constraint.
   - *Core Mechanism:* Replaces cross-entropy probability tracking with the Continuous Wasserstein-1 metric. Trains the Discriminator (now referred to as the mathematical "Critic") exactly **5 times** for every single Generative update. Evaluates gradients of interpolated images to produce gradient penalties.
3. **SAGAN (Self-Attention GAN)**
   - *Paradigm:* Allows long-range structural modeling so disconnected pixel locations (like matching eyes or hair symmetry) can directly influence each other.
   - *Core Mechanism:* Abandons `BCE` for the zero-centered `Hinge Loss`. Incorporates Spectral Normalization (`utils.spectral_norm`) on all linear projections to tightly bound singular values.
4. **USE-CMHSA-GAN (State of the Art / 2024 Context)**
   - *Paradigm:* Replaces brittle standard transposed convolutions with dense squeeze/excitation feature fusion.
   - *Core Mechanism:* Integrates Upsampling Squeeze-and-Excitation (USE) to preserve channel-wise attention, coupled tightly with Conv-based Multi-Head Self-Attention (CMHSA). Outperforms classic spatial mapping by aggressively emphasizing critical sub-feature relationships.

---

## 🛠 Architectural Challenges & Remediation (The Debugging Path)

Developing highly-sensitive adversarial environments introduces intense memory and scaling difficulties. Here is an explicit breakdown of bugs encountered, structural flaws diagnosed, and how the pipeline was mathematically secured:

### 1. The WGAN-GP PyTorch AMP Gradient Scaler Collision
- **What went wrong:** During Phase 8 validation, WGAN-GP training catastrophically failed immediately upon entering `Epoch 1`. PyTorch raised a fatal `RuntimeError: step() has already been called since the last update()`.
- **Diagnosis:** Modern deep learning pipelines demand Automatic Mixed Precision (AMP) to train effectively, converting massive `float32` tensors to `float16` and using a `GradScaler` to prevent gradients from underflowing into dead zeros. In DCGAN or SAGAN, both D and G update exactly once on a 1:1 ratio. WGAN requires a $N_{critic} = 5$ update constraint. The code originally scaled and stepped the critic 5 times inside a `for` loop, but only attempted to update PyTorch's internal dynamic scaler once at the very end of the total step.
- **How we fixed it:** Modulating the `trainers/wgan_trainer.py`. We pushed the `self.scaler.update()` method completely inside the inner 5-iteration loop. The GPU scaler now mathematically validates every instantaneous gradient scaling correction immediately after stepping the discriminator's optimizer avoiding memory corruption.

### 2. High-Capacity PyTorch Runtime Attribute Deprecation
- **What went wrong:** System threw an `AttributeError` evaluating `_C._CudaDeviceProperties`.
- **Diagnosis:** Attempting to query `total_mem` generated anomalous behavior, blocking device mapping protocols inside the `device.py` utility completely. We realized this was tied to recent aggressive API overhauls within PyTorch >2.0 configurations targeting A100/HPC nodes.
- **How we fixed it:** Rebuilt the `get_device()` method natively wrapping hardware queries dynamically using the modernized `total_memory` attribute protocol ensuring the pipeline safely maps computational graphs even in unstable SLURM instances.

### 3. Metric Evaluation Vulnerability (Inception vs. FID)
- **What went wrong:** Originally scoped to evaluate models with the classic `Inception Score (IS)`.
- **Diagnosis:** `IS` fails dramatically at penalizing real-world mode collapse on highly specialized datasets like 2D illustrations because it depends entirely on 1,000 ImageNet biological classes. Generating identical red anime eyes forever will confusingly score "perfectly" under `IS`.
- **How we fixed it:** Hard-pivoted into utilizing `clean-fid`. Instead of relying on raw biological object classification, Fréchet Inception Distance calculates the actual mathematical density curve of the generated vector representation directly against the exact real validation dataset representation vectors. This acts as an unbiased continuous topological constraint.

---

## 🚀 Execution & Usage Guide

### A. Environment Preparation
Containerize your process inside the Python virtual environment on the node prior to any hardware execution:
```bash
source venv/bin/activate
```

### B. Immediate Training Workflows
To immediately trigger a natively sequenced training cascade (DCGAN $\rightarrow$ WGAN-GP $\rightarrow$ SAGAN $\rightarrow$ USE-CMHSA-GAN):
```bash
bash scripts/train_all.sh
```

**Single Model Selection & CLI Execution:**
If you prefer precise runtime modifications, access the CLI directly targeting an individual architecture:
```bash
python3 gan_anime_faces.py train --model sagan --epochs 150 --batch-size 64
```
*(Add `--quick-test` for a rapid 2-epoch pipeline burn-in visualization).*

### C. Evaluation & Inference
The system tracks fixed-latent grids continuously inside `outputs/<model_name>/samples/` for human verification.
To initiate a complete validation suite generating the Fréchet Inception Distance (`FID`) scores on finished `.pt` weights:
```bash
bash scripts/evaluate_all.sh
```

---

## 📚 References & Literature
1. **Generative Adversarial Nets** — Goodfellow et al. (2014) *[DCGAN implementation foundation]*
2. **Wasserstein GAN** — Arjovsky, Chintala, Bottou (2017) *[Critic iterations and Wasserstein-1 distances]*
3. **Improved Training of Wasserstein GANs** — Gulrajani et al. *[Gradient Penalty scaling formulations]*
4. **Self-Attention Generative Adversarial Networks** — Zhang et al. (2018) *[Spectral Normalization dynamics]*
5. **Anime Face Generation Using Upsampling Squeeze-and-Excitation...** *(Reference SOTA 2024 Mechanics)*.
