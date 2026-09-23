# Lightweight MobileViT-Graph Network for Topological Rural Road Extraction

> **Research Project** — Satellite Imagery-based Rural Road Network Extraction using MobileViT v2 + clDice Loss

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Deployment](https://img.shields.io/badge/Deployment-Edge_Ready-blue)

---

## About

In developing nations like India, mapping millions of kilometers of rural, unpaved roads under initiatives such as the **Pradhan Mantri Gram Sadak Yojana (PMGSY)** is critical. However, current manual auditing workflows are prohibitively slow and labor-intensive.

Standard deep learning segmentation models trained on urban datasets exhibit severe **Domain Shift** when deployed in rural environments. Rural roads are slender, irregular, and frequently occluded by **tree canopies, shadows, and complex terrain**. 

### Core Objective
Construct an ultra-lightweight **encoder-decoder architecture** using **MobileViT v2** as a fast, low-parameter backbone. By synthesizing this with the graph-theoretic **clDice (Centerline-Dice) Loss**, we enforce contiguous, navigable road predictions on Indian rural topographies — all while remaining optimized for edge-device deployment.

---

## System Architecture

The pipeline consists of a comprehensive training, augmentation, inference, and topological post-processing workflow designed specifically to handle high-resolution satellite imagery while maintaining spatial connectivity.

![System Architecture](figures/fig1_system_architecture.png)

---

## Network Architecture (MobileViT v2)

Our backbone replaces the $O(N^2)$ computational complexity of standard Vision Transformers with **localized linear self-attention**. It maintains a global receptive field to "see" past occlusions where CNNs fail, while keeping the parameter footprint ultra-low.

![Network Architecture](figures/fig2_network_architecture.png)

| Model | Parameters | Size |
|---|---|---|
| Baseline U-Net | 31,037,633 | Heavy (~118 MB) |
| **MobileViT v2 (ours)** | **1,604,657** | **Ultra-lightweight (1.6M / ~6 MB)** |

---

## Key Methodologies

### 1. Topology-Preserving Loss (clDice)
We replace standard pixel-wise IoU/BCE losses with a **graph-theoretic topology-preserving loss**. It computes overlap explicitly on the morphological skeleton of predicted and ground-truth road centerlines:
```math
clDice(X, Y) = 2 \times \frac{Prec_{cl}(X, Y) \times Rec_{cl}(X, Y)}{Prec_{cl}(X, Y) + Rec_{cl}(X, Y)}
```
This mathematically penalizes topological disconnections, enforcing that predicted roads form continuous, navigable paths rather than fragmented pixel blobs.

### 2. Canopy Resilience via Data Augmentation
To combat the severe domain shift of rural environments, the training pipeline mathematically forces the model to bridge hidden roads underneath thick tree canopies through simulated occlusions.

![Canopy Augmentation](figures/fig4_augmented_canopy.png)

### 3. Aggressive Gap Bridging & Post-Processing
Our inference engine utilizes computer vision morphology and a custom topological gap-healing algorithm to dynamically bridge broken segments and extract single-pixel centerlines for routing.

![Gap Bridging](figures/fig5_graph_extraction_gap_bridging.png)

---

## Evaluation & Ablation Study

To prove the efficacy of the proposed modifications, we conducted rigorous ablation studies comparing the baseline configurations against our proposed architecture.

![Ablation Study](figures/fig7_ablation_study.png)

![Training Dynamics](figures/fig3_training_dynamics.png)

---

## Qualitative Results & Resilience Analysis

Compared to standard architectures (U-Net, DeepLabV3+), our network successfully preserves network topology even in highly occluded regions. By extracting the graph structure, we can compute metrics like **Betweenness Centrality** to identify critical infrastructure bottlenecks for disaster relief.

![Qualitative Comparison](figures/fig8_qualitative_comparison.png)

![Resilience Analysis](figures/fig6_centrality_resilience.png)

---

## Project Structure

```text
.
├── backend/                        # ML source code, training, and deployment tooling
│   ├── requirements.txt            # Full training/dev environment
│   ├── requirements-edge.txt       # Minimal on-device inference environment (for predict_onnx.py)
│   ├── scripts/
│   │   ├── train.py                # Training loop (IoU-based, collapse-gated checkpointing)
│   │   ├── export_onnx.py          # Trained checkpoint -> ONNX, with parity verification
│   │   ├── evaluate.py             # Held-out evaluation
│   │   ├── predict_single_image.py # PyTorch high-res inference with topological post-processing
│   │   └── predict_onnx.py         # On-device inference via ONNX Runtime (no PyTorch/CUDA needed)
│   └── src/
│       ├── models/                 # MobileViT v2 and U-Net architectures
│       ├── data/                   # Data ingestion and weak label processing
│       └── utils/                  # Loss functions (clDice, BCE), metrics, postprocessing
│
├── models/                         # Trained checkpoints
├── data/samples/                   # High-resolution satellite testing images
├── figures/                        # Generated diagrams and academic figures
├── results/                        # Output visualizations
├── notebooks/                      # Jupyter Notebooks for exploration
├── Final_Project_Report.pdf        # Comprehensive Technical Report
└── README.md
```

---

## Quickstart

### 1. Prerequisites
Clone the repository and install the required dependencies:
```bash
pip install -r backend/requirements.txt
```

### 2. Run Inference on a Satellite Image
You can test the topology-aware extraction on any satellite image. The script automatically handles scale mismatches and applies aggressive morphological gap-closing.
```bash
python backend/scripts/predict_single_image.py data/samples/100034_sat.jpg \
       --model models/mobilevit_v2_best.pth \
       --output results/prediction_100034.png
```

### 3. Export to ONNX (Edge Deployment)
Compile a trained checkpoint into an ONNX graph for deployment on edge devices like drones or mobile mappers. 
```bash
python backend/scripts/export_onnx.py \
       --checkpoint models/mobilevit_v2_best.pth \
       --output models/mobilevit_v2.onnx
```

### 4. Run Inference On-Device (no PyTorch required)
`backend/scripts/predict_onnx.py` is the on-device reference implementation: ONNX Runtime + OpenCV + NumPy + SciPy only. This represents what actually ships to a drone companion computer or field laptop.
```bash
pip install -r backend/requirements-edge.txt   # on the target device
python backend/scripts/predict_onnx.py data/samples/100034_sat.jpg \
       --model models/mobilevit_v2.onnx \
       --output results/prediction_100034_onnx.png
```

---

## Evaluation Metrics

| Metric | Type | Purpose |
|---|---|---|
| **clDice** | Topological | Skeleton intersection & centerline connectivity |
| **APLS** | Graph / Routing | Navigability & path-length similarity |
| **IoU / F1** | Pixel-level | Baseline spatial segmentation accuracy |
| **FPS / Latency** | Deployment | Edge-device real-world inference speed |

---

## References

- **MobileViT:** Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Mehta & Rastegari, 2021)
- **clDice:** A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation (Shit et al., 2021)
- **DeepGlobe:** Road Extraction Challenge (Demir et al., 2018)
- **PMGSY:** Pradhan Mantri Gram Sadak Yojana, Government of India
