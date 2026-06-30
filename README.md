<h1 align="center">GeoProp-GS</h1>

<p align="center">
  <strong>Geometry-Propagated Gaussian Splatting for<br>
  Aerial Sparse Novel View Synthesis</strong>
</p>

<p align="center">
  Yijing Wang · Xu Tang · Jingjing Ma · Xiangrong Zhang
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ECCV-2026-4b8bbe?style=flat-square" alt="ECCV 2026">
  <img src="https://img.shields.io/badge/Task-Aerial%20Sparse%20NVS-7c3aed?style=flat-square" alt="Aerial Sparse NVS">
  <img src="https://img.shields.io/badge/Method-3D%20Gaussian%20Splatting-f97316?style=flat-square" alt="3D Gaussian Splatting">
  <a href="https://github.com/TangXu-Group/GeoProp-GS">
    <img src="https://img.shields.io/badge/Code-GeoProp--GS-22c55e?style=flat-square" alt="Code">
  </a>
</p>

---

## News

- **[2026.07]** Initial implementation of **GeoProp-GS** is released.
- **[2026.08]** We will release:
  - benchmark datasets,
  - DGI initialization models,
  - a more comprehensive README with detailed usage instructions.

---

## Abstract

3D Gaussian Splatting achieves impressive novel view synthesis for ground-level scenes. However, its performance degrades significantly in aerial domains due to the sparse viewpoint sampling imposed by platform constraints. Existing sparse-view methods often rely on structure-from-motion initialization, which produces incomplete geometry under repetitive textures and limited viewpoint overlap.

We present **GeoProp-GS**, a method designed for aerial sparse novel view synthesis. GeoProp-GS introduces **Depth-guided Geometric Initialization (DGI)** to generate dense point clouds and extend coverage to unobserved regions, **Geometry-aware Virtual Anchors (GVA)** to provide geometric support for under-constrained regions, and **Anchor-constrained Gaussian Optimization (AGO)** to stabilize optimization through reliable anchors and learnable residuals.

Extensive experiments demonstrate that GeoProp-GS achieves state-of-the-art performance and serves as an effective plug-and-play framework for aerial sparse-view reconstruction.

**Keywords:** Aerial Photogrammetry · Aerial-view Sparse Novel View Synthesis · 3D Gaussian Splatting

> ** Upcoming Release**
>
> The benchmark datasets and DGI initialization models will be publicly released in **August 2026**. We will also provide a more comprehensive README with detailed instructions and examples.


---

## Installation

```bash
git clone https://github.com/TangXu-Group/GeoProp-GS.git
cd GeoProp-GS

conda create -n geoprop-gs python=3.8 -y
conda activate geoprop-gs

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 \
-c pytorch -c conda-forge -y

pip install plyfile tqdm matplotlib torchmetrics timm \
opencv-python imageio open3d scikit-learn scipy
```

### Submodules

This repository does not include the `submodules/` directory. Please download the required CUDA extensions from [FSGS](https://github.com/VITA-Group/FSGS) and place them under `submodules/`:

```bash
git clone https://github.com/VITA-Group/FSGS.git ../FSGS
mkdir -p submodules
cp -r ../FSGS/submodules/diff-gaussian-rasterization-confidence submodules/
cp -r ../FSGS/submodules/simple-knn submodules/

pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
```

---

## Data

The benchmark datasets and DGI initialization models are currently being prepared for public release and will be available in **August 2026**.

The current implementation expects each scene to have the following structure:

```text
scene_xxx/
  Images/
  sparse/0/
  poses_bounds.npy
  3_views/dense/fused.ply
```

For GeoProp-GS initialization, DGI additionally produces

```text
points_dense_*.ply
points_proposal_*.ply
```

The current loader contains dataset-specific path settings in

```text
scene/dataset_readers.py
```

Please update these paths for your local environment before running experiments.

---

## Quick Start

<table>
<tr>
<td width="33%" align="center">
<strong>1. Prepare Data</strong><br>
COLMAP reconstruction
</td>

<td width="33%" align="center">
<strong>2. Train</strong><br>
Optimize reliable and proposal Gaussians
</td>

<td width="33%" align="center">
<strong>3. Evaluate</strong><br>
Render novel views and compute metrics
</td>
</tr>
</table>

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_residual.py \
  --source_path /path/to/scene_xxx \
  --model_path output/geoprop-gs/scene_xxx \
  --eval \
  --n_views 3 \
  --colmap real_large_voxel_points \
  --proposal_lr_multiplier 1 \
  --proposal_dc_mean_reg 3e-4 \
  --proposal_dc_var_reg 1e-5 \
  --use_confidence
```

### Render

```bash
CUDA_VISIBLE_DEVICES=0 python render_residual.py \
  --source_path /path/to/scene_xxx \
  --model_path output/geoprop-gs/scene_xxx \
  --iteration 10000 \
  --n_views 3 \
  --colmap real_large_voxel_points
```

### Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python metrics.py \
  --source_path /path/to/scene_xxx \
  --model_path output/geoprop-gs/scene_xxx
```

For batch experiments, edit the dataset paths in

```bash
train_levir.sh
train_3das.sh
```

> **Note**
>
> A more comprehensive user guide with detailed data preparation and reproduction instructions will be released in **August 2026**.

---

## Citation

If you find this project useful, please cite

```bibtex
@inproceedings{wang2026geopropgs,
  title     = {Geometry-Propagated Gaussian Splatting for Aerial Sparse Novel View Synthesis},
  author    = {Wang, Yijing and Tang, Xu and Ma, Jingjing and Zhang, Xiangrong},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2026}
}
```

---

## Roadmap

- [x] ECCV 2026 paper
- [x] Initial source code release
- [ ] Benchmark datasets (August 2026)
- [ ] DGI initialization models (August 2026)
- [ ] Comprehensive README and documentation (August 2026)

---

## Acknowledgements

This project builds upon **3D Gaussian Splatting**, **COLMAP**, **Depth Anything V2**, **FSGS**, and diffusion-based image inpainting. We sincerely thank the authors of these excellent open-source projects.

---

## Contact

If you have any questions or suggestions, please feel free to open an issue.

For academic inquiries:

**Yijing Wang**  
yijingwang@stu.xidian.edu.cn
