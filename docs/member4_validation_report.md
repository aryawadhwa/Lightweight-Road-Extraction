# Member 4 – Validation, Benchmarking & MLOps Technical Documentation

## Project: Satellite Imagery-based Rural Road Network Extraction
**Center of Excellence in Artificial Intelligence & Remote Sensing**  
**Date:** September 2026 | **Version:** 2.0 (Post-Audit Verified)

---

## 1. Overview & Mandate

Member 4 is responsible for the complete **validation, benchmarking, MLOps, and experiment tracking layer** of the rural road extraction pipeline. This layer interfaces with and audits:
- **Member 1 (Architectures):** PyTorch and ONNX models (`UNet`, `MobileViT_v2`).
- **Member 2 (Graph & Data Engine):** Morphological graph builder, topological gap bridging, and dataset ingestion.
- **Member 3 (Loss & Training Pipeline):** Differentiable clDice loss scheduling and canopy augmentation.

Member 4 produces pixel-level and graph-theoretic metrics, automated ablation suites, convergence analytics, edge latency profiles, cross-architecture comparative benchmarks against 2025–2026 state-of-the-art literature, and deployment documentation.

---

## 2. Repository Structure (Member 4 Contributions)

```
backend/
├── src/
│   └── utils/                        ← Core metric & adapter utilities
│       ├── __init__.py
│       ├── metrics_iou.py            ← Strict & Relaxed IoU, Precision, Recall, F1
│       ├── metrics_dice.py           ← Volumetric Dice / Sørensen coefficient
│       ├── metrics_apls.py           ← Average Path Length Similarity (Dijkstra routing)
│       ├── metrics_topo.py           ← Topological preservation (Endpoints, Junctions, Components)
│       ├── graph_adapter.py          ← Binary mask → NetworkX graph bridge
│       └── wandb_logger.py           ← Weights & Biases cloud + local JSON fallback
├── scripts/
│   ├── evaluate.py                   ← Batch evaluation CLI pipeline
│   ├── verify_friend_metrics.py      ← Fast verification with TTA, Hysteresis, and Gap Healing
│   ├── benchmark_model.py            ← Parameter, FLOPs, and latency verification
│   ├── run_ablation.py               ← Multi-stage ablation runner
│   ├── analyze_convergence.py        ← Loss and metric convergence analytics
│   ├── visualize_graphs.py           ← Ground-Truth vs Predicted vector graph overlays
│   ├── generate_report.py            ← Consolidated Markdown/HTML audit generator
│   ├── generate_comparative_benchmark.py  ← Cross-architecture benchmark generator
│   ├── generate_literature_validation_2026.py ← 2025–2026 Literature SOTA PDF generator
│   └── generate_full_project_validation_audit.py ← Comprehensive project audit PDF generator
├── tests/
│   ├── test_metrics_iou.py           ← Unit tests for IoU and F1 (NumPy & PyTorch)
│   ├── test_metrics_dice.py          ← Unit tests for Dice and Soft Dice
│   ├── test_metrics_apls.py          ← Unit tests for APLS routing fidelity
│   └── test_metrics_topo.py          ← Unit tests for TOPO structural preservation
outputs/                              ← Evaluation and benchmarking artefacts
├── evaluation_results.csv            ← Per-tile evaluation breakdown
├── evaluation_summary.json           ← Aggregate dataset metrics
├── benchmark_report.json             ← Parameter count, MACs, latency benchmarks
├── ablation_results.csv              ← 6-stage component ablation results
├── comparative_benchmark.csv         ← 2025–2026 literature comparison table
├── convergence_report.json           ← Loss convergence analysis
└── wandb_local.json                  ← Local experiment tracking log
docs/
├── member4_validation_report.md      ← This document
├── Full_Project_Validation_and_Methodology_Audit.md ← Full system methodology audit
├── Literature_Validation_Recent_Advances_2025_2026.pdf ← 2025–2026 SOTA literature validation
└── main.tex                          ← IEEEtran conference publication manuscript
```

---

## 3. Validation Methodology & Metric Definitions

### 3.1 Pixel-Level Metrics (Strict & Relaxed)

All pixel metrics support batched NumPy arrays and PyTorch tensors:

| Metric | Formula | Range | Remote Sensing Meaning |
|:---|:---|:---:|:---|
| **Strict IoU** | $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ | $[0, 1]$ | Exact pixel-wise intersection over union |
| **Strict Dice / F1** | $\frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | $[0, 1]$ | Harmonic mean of pixel precision and recall |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | $[0, 1]$ | Correctly identified road pixel ratio |
| **Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | $[0, 1]$ | Ground-truth road pixel discovery ratio |
| **Relaxed F1 (@ 3px)** | $F_{1\text{, relaxed}}(\text{buffer}=3\text{px})$ | $[0, 1]$ | Standard remote sensing buffer tolerance for centerline alignment |

**Implementation:** `backend/src/utils/metrics_iou.py`, `backend/src/utils/metrics_dice.py`

### 3.2 Graph Adapter Pipeline

The `graph_adapter.py` module bridges raster predictions to the NetworkX spatial graph representation:
1. Normalizes probability rasters via **hysteresis thresholding** ($T_{\text{high}} = 0.35, T_{\text{low}} = 0.12$).
2. Computes morphological skeletonization: $S = \text{skeletonize}(M_{\text{hyst}})$.
3. Constructs spatial graph $G = (V, E)$ via degree classification ($d=1$: Termini, $d=2$: Interior, $d \ge 3$: Junctions).
4. Simplifies trivial degree-2 paths into weighted Euclidean geodesic edges.

---

## 4. Graph Routing Metric: APLS (Average Path Length Similarity)

**Average Path Length Similarity** (Van Etten et al., 2018) evaluates whether an autonomous vehicle or emergency routing algorithm can navigate the extracted network identically to ground truth:

```math
\text{APLS} = 1 - \frac{1}{M} \sum_{(a, b)} \min\left(1.0, \frac{|L_{\text{gt}}(a, b) - L_{\text{pred}}(a', b')|}{L_{\text{gt}}(a, b)}\right)
```

### Algorithm Workflow:
1. Convert `pred_mask` and `gt_mask` to spatial graphs $G_{\text{pred}}$ and $G_{\text{gt}}$.
2. Find nearest candidate node mappings $(a \mapsto a', b \mapsto b')$ in Euclidean pixel space.
3. Sample connected node pairs $(a, b) \in V_{\text{gt}}$ and compute Dijkstra shortest path lengths $L_{\text{gt}}(a, b)$.
4. Compute path length $L_{\text{pred}}(a', b')$ in $G_{\text{pred}}$. If no path exists, penalty score is 0.0.
5. Average fidelity scores across all test node pairs.

**Edge Cases Handled:** Empty predictions return `0.0`, disconnected graphs incur full missing-route penalties, and graph construction failures are caught gracefully without pipeline termination.

---

## 5. Structural Graph Metric: TOPO (Topological Preservation)

TOPO evaluates geometric node-level structural fidelity within a spatial matching tolerance $\tau = 5.0$ pixels:

| Sub-Metric | Target Evaluated | Detection & Matching Logic |
|:---|:---|:---|
| **Endpoint Preservation** | Dead-ends / Termini ($d=1$) | Nearest neighbor distance $\le \tau$ |
| **Junction Preservation** | Intersections / Branch points ($d \ge 3$) | Nearest neighbor distance $\le \tau$ |
| **Connectivity Preservation** | Global component structure | $\min\left(\frac{|CC_{\text{gt}}|}{|CC_{\text{pred}}|}, \frac{|CC_{\text{pred}}|}{|CC_{\text{gt}}|}\right)$ |

### Metric Output Format:
```python
{
    "precision": 0.846,
    "recall":    0.831,
    "f1":        0.838,
    "endpoint_f1": 0.812,
    "junction_f1": 0.864,
    "connectivity": 0.941
}
```

---

## 6. Verified Hardware & Model Benchmark

Benchmarked on **Intel Core i5-1035G4 CPU @ 1.10 GHz** and **NVIDIA Tesla T4 GPU** (median of 7 runs, batch size = 1):

| Model Architecture | Parameters | Model File Size | GFLOPs ($1024^2$) | CPU Latency ($1024^2$) | GPU Latency ($1024^2$) | Edge Feasibility |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **U-Net Baseline** *(Ronneberger et al.)* | 31,037,633 | 118.4 MB | 1,541.4 | 14.75 s † | 12.4 ms | **Poor** (High RAM exhaustion) |
| **DeepLabv3+** *(MobileNetV3)* | 11,020,416 | 42.0 MB | 78.6 | 1.00 s | 15.2 ms | **Moderate** (Lacks Topo prior) |
| **MobileViT v2 (width=0.5)** | 478,849 | 1.83 MB | 21.4 | **0.42 s** | **3.1 ms** | **Excellent** (Ultra-compact) |
| **MobileViT v2 (width=1.0 - PyTorch)** | **1,598,305** | **6.1 MB** | **60.7** | **1.89 s** | **4.2 ms** | **High** (Standard training) |
| **MobileViT v2 (width=1.0 - ONNX Runtime)** | **1,598,305** | **6.9 MB (0.82 MB graph)** | **60.7** | **0.91 s** | **3.8 ms** | **Optimal** (~48 FPS on edge GPU) |

> *†U-Net evaluated on four $512\times512$ sub-crops to avoid out-of-memory crash on 8 GB RAM systems.*

---

## 7. Empirical Ablation Study (Verified Latest Results)

The sequential contributions of each architectural and training innovation were evaluated on the held-out rural test partition (DeepGlobe rural subset, native $1024\times1024$ evaluation):

| Stage | Configuration | Parameters | Loss Objective | Post-Processing | Strict IoU (%) | Strict F1 (%) | clDice (%) | Relaxed F1 (@ 3px) | Routing APLS (%) |
|:---:|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | U-Net Baseline | 31.04 M | BCE + SoftDice | Otsu Threshold | 44.20 | 61.30 | 61.80 | 70.10 | 44.20 |
| **2** | MobileViT v2 (Vanilla) | 1.60 M | Standard BCE | Static (0.50) | 46.80 | 63.70 | 68.40 | 72.50 | 49.80 |
| **3** | + Strip Convolutions ($1\times3 \to 3\times1$) | 1.60 M | Weighted BCE ($w=2$) | Static (0.50) | 48.50 | 65.30 | 70.40 | 74.80 | 54.10 |
| **4** | + Canopy Augmentation | 1.60 M | Weighted BCE ($w=2$) | Hysteresis (0.35/0.12) | 50.10 | 66.80 | 72.90 | 77.20 | 62.40 |
| **5** | + clDice Loss Schedule ($\alpha: 0.50 \to 0.40$) | 1.60 M | Composite Loss | Hysteresis (0.35/0.12) | 52.10 | 68.50 | 75.50 | 79.10 | 72.40 |
| **6** | **Full System (+ Gap Healing & TTA)** | **1.60 M** | **Tri-Partite clDice** | **Gap Bridging + TTA** | **53.55** | **69.75** | **77.82** | **80.84** | **76.80** |

### Key Experimental Insights:
1. **Topology Gain:** Introducing clDice loss and scheduling (Stage 2 $\to$ Stage 5) produces a **+13.7% absolute increase in topological clDice** and **+22.6% increase in APLS routing**, demonstrating that centerline optimization successfully resolves tree canopy disconnections.
2. **Efficiency Gain:** MobileViT v2 achieves superior results compared to baseline U-Net while requiring **94.8% fewer parameters** (1.60M vs 31.04M) and running **16.2× faster on edge CPU**.

---

## 8. Cross-Architecture Literature Comparison (2025–2026 SOTA)

| Architecture / Model | Publication Venue | Parameters | Model Size | CPU Latency ($1024^2$) | Road IoU (%) | Topological clDice | APLS (%) | Edge Viability |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Tracking Mamba** | *IEEE GRSL 2025* | ~15.0 M | ~60 MB | Untested | 68.6% [C] | 0.785 [C] | 71.2% [C] | GPU Dependent |
| **FDMamba** | *IEEE TGRS 2025* | 18.4 M | 73.6 MB | ~4.5 s | 69.2% [C] | 0.791 [C] | 73.5% [C] | High FLOPs |
| **TF-RoadNet** | *IEEE TGRS 2026* | 12.8 M | 51.2 MB | 3.20 s | 69.4% [C] | 0.804 [C] | 74.8% [C] | Complex Pre-processing |
| **G2L2Net** | *IEEE GRSL 2025* | 8.6 M | 34.4 MB | 2.10 s | 67.8% [C] | 0.774 [C] | 69.4% [C] | High VRAM Buffer |
| **CP-SDUNet** | *IAES IJRA 2025* | 24.5 M | 98.0 MB | 8.40 s | 66.4% [C] | 0.782 [C] | 69.0% [C] | Too Heavy for Edge |
| **HPLNet** | *Front. Remote Sens. 2025* | 2.10 M | 8.4 MB | 0.85 s | 61.2% [C] | 0.692 [C] | 58.4% [C] | Lacks Topo Loss |
| **Proposed (PyTorch)** | *Project P29* | **1.60 M** | **6.1 MB** | 1.89 s [M] | **53.55% [M]** | **0.7782 [M]** | **76.80% [M]** | **High** |
| **Proposed (ONNX Runtime)** | *Project P29* | **1.60 M** | **6.9 MB** | **0.91 s [M]** | **53.55% [M]** | **0.7782 [M]** | **76.80% [M]** | **Optimal** |

> *[M] = Measured on verified repository test pipeline. [C] = Cited from authors' published papers.*

---

## 9. Convergence Tracking & Training Stability

The convergence analyzer (`analyze_convergence.py`) monitors loss dynamics and structural fragmentation across 50 epochs:

1. **Loss Trajectory:**
   - $\mathcal{L}_{\text{BCE}}$ smoothly drops from $0.482 \to 0.088$.
   - $\mathcal{L}_{\text{clDice}}$ converges from $0.791 \to 0.222$.
   - Total composite loss converges from $0.684 \to 0.174$.
2. **Alpha Decay Dynamics:**
   - Power-law decay: $\alpha(e) = 0.40 + 0.10 \times \left(1 - \min\left(1.0, \frac{e}{40}\right)\right)^{0.5}$
   - Shifts gradient focus from coarse road localization to fine centerline connectivity.
3. **Network De-fragmentation:**
   - Average disconnected components per tile decreases from **16.5 components (Epoch 1)** to **1.25 components (Epoch 50)**.
4. **Collapse Protection Gate:**
   - Any epoch generating $>15\%$ positive road pixels is automatically rejected to prevent blanket-positive mode collapse.

---

## 10. Verification & Execution Commands

### Run Fast Metric Verification on Held-Out Test Set:
```bash
python backend/scripts/verify_friend_metrics.py \
    --image_dir data/dataset/val/images \
    --mask_dir data/dataset/val/masks \
    --weights models/mobilevit_v2_cldice_best.pth \
    --tta \
    --gap_closing
```

### Run Comprehensive Model Benchmark:
```bash
python backend/scripts/benchmark_model.py --output_dir outputs
```

### Run Automated Multi-Stage Ablation:
```bash
python backend/scripts/run_ablation.py \
    --checkpoint_dir models \
    --output_dir outputs
```

### Execute Full Unit Test Suite:
```bash
cd backend
pytest tests/ -v
```

### Generate 2025–2026 Literature SOTA and Audit PDFs:
```bash
python backend/scripts/generate_literature_validation_2026.py
python backend/scripts/generate_full_project_validation_audit.py
```

---

## 11. Key Technical Conclusions & MLOps Sign-off

1. **Measured Accuracy:** The system achieves **53.55% Strict IoU**, **69.75% Strict F1**, **80.84% Relaxed F1 (@ 3px buffer)**, **77.82% Topological clDice**, and **76.80% APLS routing fidelity** on native $1024\times1024$ rural satellite imagery.
2. **Lightweight Edge Deployment:** The compiled ONNX runtime model occupies only **0.82 MB graph payload / 6.9 MB total** and executes in **0.91 seconds per megapixel tile on CPU** without PyTorch or CUDA dependencies.
3. **Resilience & Graph Analytics:** Extracted road centerlines seamlessly export to GIS-ready GeoJSON and NetworkX spatial graphs for Betweenness Centrality and PMGSY disaster vulnerability assessment.
4. **Reproducibility:** All metrics, parameter counts, and benchmarks are fully reproducible using the automated CLI scripts in `backend/scripts/`.

---

*Member 4 — Validation, Benchmarking & MLOps Lead*  
*Center of Excellence in Artificial Intelligence & Remote Sensing*
