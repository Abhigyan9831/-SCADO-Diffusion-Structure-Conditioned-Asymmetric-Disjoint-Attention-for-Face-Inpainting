<div align="center">

# SCADO-Diffusion: Structure-Conditioned Asymmetric Disjoint Attention for Face Inpainting

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Diffusion](https://img.shields.io/badge/Model-Conditional%20DDPM-8A2BE2.svg)](#architecture)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary%20%2F%20Restricted-red.svg)](#intellectual-property--access-notice)

<p align="center">
  <b>A facial inpainting framework combining Denoising Diffusion Probabilistic Models (DDPM) with Structure-Conditioned Asymmetric Disjoint Attention (SCADO) and dynamic Graph Attention landmark guidance.</b>
</p>

---


---

</div>

##  Qualitative Results & Progression

<p align="center">
  Each image displays the evaluation triplet generated directly during training:
  <br>
  <b>[ Top: Ground Truth ] &nbsp;•&nbsp; [ Middle: Masked Input ] &nbsp;•&nbsp; [ Bottom: SCADO Inpainting Result ]</b>
</p>

<div align="center">

### Converged Inpainting Benchmarks (Steps 325,000 – 327,300)

| Sample 1 (Step 327.3k) | Sample 2 (Step 327.1k) | Sample 3 (Step 326.9k) | Sample 4 (Step 326.7k) | Sample 5 (Step 326.5k) |
| :---: | :---: | :---: | :---: | :---: |
| <img src="assets/step_0327300.png" width="180px" alt="Sample 327300"/> | <img src="assets/step_0327100.png" width="180px" alt="Sample 327100"/> | <img src="assets/step_0326900.png" width="180px" alt="Sample 326900"/> | <img src="assets/step_0326700.png" width="180px" alt="Sample 326700"/> | <img src="assets/step_0326500.png" width="180px" alt="Sample 326500"/> |
| <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> |

<br>

| Sample 6 (Step 326.3k) | Sample 7 (Step 326.0k) | Sample 8 (Step 325.7k) | Sample 9 (Step 325.5k) | Sample 10 (Step 325.0k) |
| :---: | :---: | :---: | :---: | :---: |
| <img src="assets/step_0326300.png" width="180px" alt="Sample 326300"/> | <img src="assets/step_0326000.png" width="180px" alt="Sample 326000"/> | <img src="assets/step_0325700.png" width="180px" alt="Sample 325700"/> | <img src="assets/step_0325500.png" width="180px" alt="Sample 325500"/> | <img src="assets/step_0325000.png" width="180px" alt="Sample 325000"/> |
| <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b>Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b>Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b>Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b>Our Model:</b> SCADO Reconst.</sub> | <sub><b>Top:</b> Ground Truth<br><b>Mid:</b> Masked<br><b> Our Model:</b> SCADO Reconst.</sub> |

</div>

<br>

---


### Architectural Mechanics:

1. **Disjoint Spatial Partitioning:**

   Spatial feature maps <img src="https://latex.codecogs.com/svg.image?f\in\mathbb{R}^{C\times%20H\times%20W}" align="center"/> are divided into corrupted positions <img src="https://latex.codecogs.com/svg.image?\Omega_m=\{i\mid%20M_i=1\}" align="center"/> and uncorrupted visible tokens <img src="https://latex.codecogs.com/svg.image?\Omega_v=\{i\mid%20M_i=0\}" align="center"/>.

2. **Asymmetric Disjoint Cross-Attention:**

   - **Queries (<img src="https://latex.codecogs.com/svg.image?Q" align="center"/>):** Projected solely from corrupted regions <img src="https://latex.codecogs.com/svg.image?\Omega_m" align="center"/>:
     
     <p align="center"><img src="https://latex.codecogs.com/svg.image?Q=W_Q\cdot%20f[\Omega_m]\in\mathbb{R}^{|\Omega_m|\times%20d}" alt="Q equation"/></p>

   - **Keys & Values (<img src="https://latex.codecogs.com/svg.image?K,V" align="center"/>):** Extracted purely from uncorrupted visible regions <img src="https://latex.codecogs.com/svg.image?\Omega_v" align="center"/>:
     
     <p align="center"><img src="https://latex.codecogs.com/svg.image?K=W_K\cdot%20f[\Omega_v],\quad%20V=W_V\cdot%20f[\Omega_v]\in\mathbb{R}^{|\Omega_v|\times%20d}" alt="K and V equation"/></p>

3. **Graph Attention Network (GAT) Landmark Prior:**

   Seven key facial landmarks generate 2D Gaussian heatmaps <img src="https://latex.codecogs.com/svg.image?H_k" align="center"/>, dynamically weighted by a Graph Attention Network (<img src="https://latex.codecogs.com/svg.image?\alpha_k" align="center"/>), producing a structured spatial cross-attention bias:

   <p align="center">
     <img src="https://latex.codecogs.com/svg.image?\tilde{A}=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}&plus;\sum_{k=1}^K\alpha_k%20H_k(i)H_k(j)\right)" alt="SCADO Attention Equation"/>
   </p>

4. **Bilateral Facial Symmetry Constraint:**

   A self-supervised symmetry loss directly penalizes horizontal asymmetry in the SCADO attention map:

   <p align="center">
     <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{sym}}=\sum_i\left\|A_i-A_{\text{mirror}(i)}\right\|^2" alt="Symmetry Loss Equation"/>
   </p>

---

## Multi-Objective Loss Formulation

The complete training objective combines generative diffusion, structural constraints, and perceptual realism:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{recon}}&plus;\lambda_{\text{sym}}\mathcal{L}_{\text{sym}}&plus;\lambda_{\text{perceptual}}\mathcal{L}_{\text{perceptual}}&plus;\lambda_{\text{adv}}\mathcal{L}_{\text{adv}}" alt="Total Loss Equation"/>
</p>

Where:

- **Reconstruction Loss (<img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{recon}}" align="center"/>):** Composite error consisting of DDPM noise prediction MSE and image-space L1 loss:
  <p align="center">
    <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{recon}}=\|\epsilon-\epsilon_\theta(x_t,t)\|^2&plus;\|x_0-\hat{x}_0\|_1" alt="Recon Loss Equation"/>
  </p>
- **Symmetry Loss (<img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{sym}}" align="center"/>):** Direct horizontal mirror penalty computed over SCADO cross-attention weights.
- **Perceptual Loss (<img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{perceptual}}" align="center"/>):** Multi-scale feature matching loss across frozen VGG-16 layers (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`).
- **Adversarial Loss (<img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{adv}}" align="center"/>):** Hinge adversarial loss via a multi-scale PatchGAN discriminator enforcing high-frequency edge sharpness.
