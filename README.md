# Anime Face Generation — Advanced GAN Benchmarking

## 📌 Project Overview
This project presents an end-to-end framework for generating high-fidelity anime faces using advanced Generative Adversarial Networks (GANs). The pipeline is built with **PyTorch** and adheres to **SOLID** design principles to ensure modularity, scalability, and clean scientific experimentation.

The project benchmarks four distinct architectures against the **Kaggle Anime Face Dataset (~63k images)**, evaluating them using state-of-the-art metrics like **Fréchet Inception Distance (FID)**.

---

## 🚀 Key Results & Performance

We evaluated all models after 200 epochs of training on 64x64 images. The scores below represents the **Fréchet Inception Distance (FID)**, where a lower score indicates higher image quality and better distribution alignment with real anime faces.

| Architecture | FID Score (Lower is better) | Assessment |
| :--- | :---: | :--- |
| **DCGAN** (Baseline) | 161.48 | Decent baseline, prone to blurring. |
| **WGAN-GP** | **38.68** | 🏆 **Best Quality & Stability.** |
| **SAGAN** | 51.35 | High diversity, slightly more artifacts than WGAN. |
| **USE-CMHSA-GAN** | 163.23 | Experimental (Needs more tuning). |

---

## 📈 Visual Results & Monitoring

For each model, we have generated high-resolution visualizations of the training process:

*   **Loss Curves**: View the Generator vs Discriminator training stability in `outputs/<model>/reports/loss_curves.png`.
*   **Image Samples**: Progression grids of generated faces across epochs are stored in `outputs/<model>/samples/`.
*   **TensorBoard Logs**: Full interactive telemetry can be viewed by running `tensorboard --logdir outputs/`.

> [!IMPORTANT]
> **WGAN-GP** emerged as the clear winner, achieving the best visual clarity and the lowest FID score by a significant margin. This highlights the effectiveness of the Wasserstein distance in modeling the complex distributions of anime art.

---

## 🧠 Architectural Deep-Dive

### 1. DCGAN (Baseline)
The foundational Deep Convolutional GAN using strided convolutions, BatchNorm, and ReLU. It serves as our anchor for evaluating architectural improvements.

### 2. WGAN-GP (Wasserstein GAN with Gradient Penalty)
Implements the Wasserstein-1 distance to provide a more meaningful loss metric. We use **Gradient Penalty** instead of weight clipping to enforce the Lipschitz constraint, resulting in significantly higher training stability and a higher quality latent-to-image mapping.

### 3. SAGAN (Self-Attention GAN)
Incorporates non-local attention mechanisms. This allows the model to capture long-range dependencies (e.g., matching the color and shape of both eyes simultaneously), which is critical for facial symmetry. It also utilizes **Spectral Normalization** on all layers to stabilize the discriminator.

### 4. USE-CMHSA-GAN (2024 Research Implementation)
An advanced variant integrating **Upsampling Squeeze-and-Excitation (USE)** blocks and **Convolutional Multi-Head Self-Attention (CMHSA)** modules. Designed to focus on specific facial features (hair, eyes, chin) during the upsampling process.

---

## 🛠 Engineering & Debugging Log (What Went Wrong & How We Fixed It)

Adversarial training is notoriously sensitive. Below are the key engineering challenges we overcame during development:

### ⚙️ 1. Hardware Property Deprecation
*   **Issue:** The system crashed when querying GPU VRAM using `torch.cuda.get_device_properties().total_mem`.
*   **Fix:** Updated the device utility (`utils/device.py`) to use the modernized `total_memory` attribute, ensuring compatibility with the latest PyTorch runtime on HPC nodes.

### 📉 2. WGAN-GP GradScaler Sync Error
*   **Issue:** PyTorch AMP (Mixed Precision) threw a `RuntimeError` because `scaler.step()` was called multiple times inside the critic loop without an intervening `scaler.update()`.
*   **Fix:** Relocated the `scaler.update()` call inside the $N_{critic}$ loop in `trainers/wgan_trainer.py`. This ensures the dynamic loss scaling is validated after every discriminator update, preventing gradient corruption.

### 📏 3. FID Evaluation Shape Mismatch
*   **Issue:** The evaluation suite failed with `Expected 4D input but got 2D` during FID calculation.
*   **Fix:** Enhanced the FID wrapper in `evaluation/fid.py` to automatically detect 2D latent batches and unsqueeze them to the required 4D shape `(N, C, 1, 1)` before passing them to the generator.

---

## 🛠 How to Use the Pipeline

### 1. Environment Setup
```bash
# Activate the pre-configured virtual environment
source venv/bin/activate
```

### 2. Automated Training
To train all four architectures consecutively:
```bash
bash scripts/train_all.sh
```

### 3. Comprehensive Evaluation
To compute FID scores for all trained models:
```bash
bash scripts/evaluate_all.sh
```

---

## 📂 Code Structure
*   `configs/`: Hyperparameter settings for each architecture.
*   `data/`: Preprocessing, augmentation, and dataset logic.
*   `models/`: Modular implementation of G and D networks.
*   `trainers/`: Template-based training loops.
*   `evaluation/`: FID and IS metric computation.
*   `visualization/`: Loss plots and latent space interpolation scripts.

---

## 📚 References
- Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (2016).
- Gulrajani et al., *Improved Training of Wasserstein GANs* (2017).
- Zhang et al., *Self-Attention Generative Adversarial Networks* (2018).
