# GCM-Seg: Ground Fissure Detection in Mining Areas

This repository provides the implementation of **GCM-Seg**, a deep learning–based semantic segmentation framework for automatic ground fissure detection in mining areas using high-resolution remote sensing imagery.

---

## Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- CUDA 11.8  
- Operating System: Windows

Dependencies can be installed according to the project configuration files.

---

## Datasets

### 1. Public Dataset

To validate the generalization capability of the proposed method, experiments were conducted on  publicly available fissure/crack segmentation dataset:

---

### 2. BLG-MFD Sub-dataset (For Reference Only)

A subset of the **BLG-MFD (Bulianta Gully Mining Field Dataset)** is provided **for reference and reproducibility purposes only**.

- Data source: high-resolution UAV orthophotos from a mining area
- Content: ground fissure semantic segmentation samples
- Purpose: qualitative demonstration and partial quantitative verification
- Usage limitation: **non-commercial research use only**

The full BLG-MFD dataset cannot be fully released due to data management and project constraints.

#### Data availability
A reference subset of BLG-MFD is archived on Zenodo:

- **Zenodo DOI**: https://doi.org/10.5281/zenodo.18041897  
- **Status**: *Under review / pending approval*

Once the Zenodo record is fully published, the link will be updated accordingly.

---

## Project Structure

