# rt-games Metrics Reference

This document lists all evaluation metrics implemented in the rt-games toolkit, their paper sources, and the dependent models required for computation.

---

## Quick Reference Table

| Metric | Category | Model Dependency | Paper Source |
|--------|----------|------------------|--------------|
| LPIPS | Content Preservation | AlexNet/VGG (pretrained) | Zhang et al., CVPR 2018 |
| SSIM | Content Preservation | None (pixel-based) | Wang et al., TIP 2004 |
| Content Loss | Content Preservation | VGG16 (ImageNet) | Gatys et al., CVPR 2016 |
| Gram Loss | Style Fidelity | VGG16 (ImageNet) | Gatys et al., CVPR 2016 |
| FID | Style Fidelity | art_inception / InceptionV3 | Heusel et al., NeurIPS 2017 |
| SIFID | Style Fidelity | art_inception / InceptionV3 | Shaham et al., ICCV 2019 |
| HistoGAN | Style Fidelity | None (histogram-based) | Afifi et al., CVPR 2021 |
| CFSD | Content Structure | VGG16 (ImageNet) | Chung et al. (StyleID), 2024 |
| ArtFID | Composite | AlexNet + art_inception | Wright & Kummerer, 2022 |
| Warping Error | Temporal Coherence | RAFT (optical flow) | Ruder et al., GCPR 2016 |
| Temporal LPIPS | Temporal Coherence | RAFT + AlexNet | Ioannou & Maddock, ToG 2024 |
| Depth Error | Depth Preservation | MiDaS (DPT_Hybrid) | Liu et al., NPAR 2017 |

---

## Content Preservation Metrics

### LPIPS (Learned Perceptual Image Patch Similarity)

**Purpose:** Measures perceptual similarity between content and stylized images.

**Paper:**
> Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
> - arXiv: https://arxiv.org/abs/1801.03924
> - Code: https://github.com/richzhang/PerceptualSimilarity

**Model Dependency:**
- `lpips` library with AlexNet backbone (default) or VGG
- Pretrained weights from the original LPIPS release

**Implementation:** `rt_games/metrics/perceptual.py::lpips_content()`

**Used in papers:** paper.md (G-buffer NST), StyleID, MambaST, AttenST

---

### SSIM (Structural Similarity Index)

**Purpose:** Measures structural similarity between content and stylized images.

**Paper:**
> Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing, 13(4), 600-612.

**Model Dependency:**
- None (purely pixel-based computation)
- Uses `piq` library implementation

**Implementation:** `rt_games/metrics/perceptual.py::ssim_score()`

**Used in papers:** paper.md (G-buffer NST)

---

### Content Loss (VGG Feature MSE)

**Purpose:** Measures content preservation via deep feature similarity.

**Paper:**
> Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.
> - arXiv: https://arxiv.org/abs/1508.06576

**Model Dependency:**
- **VGG16** pretrained on ImageNet (IMAGENET1K_V1)
- Features extracted from `relu2_2` layer

**Implementation:** `rt_games/metrics/perceptual.py::content_loss()`

**Used in papers:** paper.md (G-buffer NST)

---

## Style Fidelity Metrics

### Gram Loss (Style Loss)

**Purpose:** Measures style similarity via Gram matrix comparison of VGG features.

**Paper:**
> Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.

**Model Dependency:**
- **VGG16** pretrained on ImageNet
- Gram matrices computed from layers: `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`

**Implementation:** `rt_games/metrics/style.py::gram_loss()`

**Used in papers:** paper.md (G-buffer NST), StyleID, MambaST

---

### FID (Fréchet Inception Distance) for Style

**Purpose:** Measures distribution similarity between style images and stylized outputs.

**Papers:**
> Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS 2017.

For style transfer, uses **art_inception**:
> Wright, M., & Kummerer, M. (2022). *ArtFID: A New Metric for Artistic Style Transfer.*
> - Checkpoint: https://huggingface.co/matthias-wright/art_inception

**Model Dependency:**
- **art_inception** (InceptionV3 fine-tuned on art datasets) — *recommended for SOTA comparability*
- Or standard **InceptionV3** (ImageNet)

**Implementation:** `rt_games/metrics/style.py::fid_score()`

**Used in papers:** StyleID, MambaST, AttenST (all use art_inception)

---

### SIFID (Single Image FID)

**Purpose:** Per-image FID between style reference and stylized output. Useful for single-style evaluation.

**Paper:**
> Shaham, T. R., Dekel, T., & Michaeli, T. (2019). *SinGAN: Learning a Generative Model from a Single Natural Image.* ICCV 2019.
> - arXiv: https://arxiv.org/abs/1905.01164

**Model Dependency:**
- **art_inception** or **InceptionV3**
- Computes Fréchet distance per image pair, then averages

**Implementation:** `rt_games/metrics/style.py::sifid_score()`

**Used in papers:** paper.md (G-buffer NST)

---

### HistoGAN Distance

**Purpose:** Measures color distribution similarity via RGB-uv histogram matching.

**Paper:**
> Afifi, M., Brubaker, M. A., & Brown, M. S. (2021). *HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms.* CVPR 2021.
> - arXiv: https://arxiv.org/abs/2011.11731

**Model Dependency:**
- None (histogram-based, differentiable soft binning)

**Implementation:** `rt_games/metrics/style.py::histogan_distance()` using `RGBuvHistBlock`

**Used in papers:** StyleID, MambaST, AttenST

---

### CFSD (Content Feature Structural Distance)

**Purpose:** Measures structural preservation via patch-wise cosine similarity of VGG features.

**Paper:**
> Chung, J., et al. (2024). *StyleID: Identity-Preserving Style Transfer with Diffusion Models.*
> - Uses patch self-similarity from VGG conv3 features

**Model Dependency:**
- **VGG16** pretrained on ImageNet
- Features from `relu3_3`, patch unfolding with cosine similarity

**Implementation:** `rt_games/metrics/style.py::cfsd()`

**Used in papers:** StyleID, MambaST, AttenST

---

## Composite Metrics

### ArtFID

**Purpose:** Combined metric balancing content preservation and style fidelity.

**Definition:**
```
ArtFID = (1 + LPIPS_content) × (1 + FID_style)
```

**Paper:**
> Wright, M., & Kummerer, M. (2022). *ArtFID: A New Metric for Artistic Style Transfer.*
> - GitHub: https://github.com/matthias-wright/art-fid

**Model Dependency:**
- **AlexNet** (for LPIPS)
- **art_inception** (for FID)

**Implementation:** `rt_games/metrics/composite.py::artfid()`

**Used in papers:** StyleID, MambaST, AttenST

---

## Temporal / Video Metrics

### Warping Error

**Purpose:** Measures temporal consistency by warping frame t to t+1 using optical flow and comparing with actual frame t+1.

**Paper:**
> Ruder, M., Dosovitskiy, A., & Brox, T. (2016). *Artistic Style Transfer for Videos.* GCPR 2016.
> - arXiv: https://arxiv.org/abs/1604.08610

**Model Dependency:**
- **RAFT** optical flow model (or precomputed flow from game engine)
  - Model: `raft-small` from `princeton-vl/RAFT`
  - Paper: Teed & Deng, ECCV 2020

**Implementation:** `rt_games/metrics/temporal.py::warping_error()`

**Used in papers:** paper.md (G-buffer NST)

---

### Temporal LPIPS

**Purpose:** Perceptual temporal consistency — LPIPS between warped frame t and actual frame t+1.

**Paper:**
> Ioannou, E., & Maddock, S. (2024). *Towards Real-time G-buffer-Guided Style Transfer in Computer Games.* IEEE Transactions on Games.

**Model Dependency:**
- **RAFT** for optical flow
- **AlexNet** (LPIPS backbone)

**Implementation:** `rt_games/metrics/temporal.py::temporal_lpips()`

**Used in papers:** paper.md (G-buffer NST)

---

### Depth Error

**Purpose:** Measures depth preservation between original and stylized frames.

**Paper:**
> Liu, X.-C., Cheng, M.-M., Lai, Y.-K., & Rosin, P. L. (2017). *Depth-aware Neural Style Transfer.* NPAR 2017.

Uses MiDaS for depth estimation:
> Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer.* TPAMI.

**Model Dependency:**
- **MiDaS** depth estimation (DPT_Hybrid or DPT_Large)
  - Hub: `intel-isl/MiDaS`
- Or precomputed depth maps from game engine

**Computation:** Scale-invariant log RMSE between depth maps.

**Implementation:** `rt_games/metrics/depth.py::depth_error()`

**Used in papers:** paper.md (G-buffer NST)

---

## Model Summary

| Model | Source | Used For | Download |
|-------|--------|----------|----------|
| **VGG16** | torchvision | Gram loss, Content loss, CFSD | Auto (ImageNet weights) |
| **AlexNet** | lpips library | LPIPS, Temporal LPIPS | Auto (LPIPS weights) |
| **art_inception** | HuggingFace | FID, SIFID, ArtFID | [art_inception.pth](https://huggingface.co/matthias-wright/art_inception) |
| **InceptionV3** | torchvision | FID (standard), SIFID | Auto (ImageNet weights) |
| **RAFT** | torch.hub | Warping Error, Temporal LPIPS | `princeton-vl/RAFT` |
| **MiDaS** | torch.hub | Depth Error | `intel-isl/MiDaS` |

---

## References

1. **Gatys et al. (2016)** — Neural Style Transfer foundational paper
2. **Zhang et al. (2018)** — LPIPS perceptual metric
3. **Wang et al. (2004)** — SSIM structural similarity
4. **Heusel et al. (2017)** — FID metric
5. **Shaham et al. (2019)** — SIFID / SinGAN
6. **Afifi et al. (2021)** — HistoGAN color histograms
7. **Wright & Kummerer (2022)** — ArtFID composite metric
8. **Ruder et al. (2016)** — Video style transfer / warping error
9. **Ioannou & Maddock (2024)** — G-buffer guided NST for games
10. **Liu et al. (2017)** — Depth-aware NST
11. **Ranftl et al. (2020)** — MiDaS depth estimation
12. **Teed & Deng (2020)** — RAFT optical flow

---

## Usage Notes

### For SOTA Comparability
- Use `art_inception` for FID/SIFID (set `use_art_inception=True`, which is the default)
- This matches StyleID, MambaST, AttenST evaluation protocols

### For Game/Video Evaluation
- Provide precomputed optical flow from game engine (Unity motion vectors) to avoid RAFT overhead
- Provide engine depth maps to skip MiDaS computation
- Export frames in consistent color space (sRGB after tone mapping)

### Model Caching
All heavy models (VGG, Inception, MiDaS, RAFT) are cached via `ModelCache` to avoid reloading during batch evaluation.


