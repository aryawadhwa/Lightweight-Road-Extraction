# Red-Team Review Report

## REVIEWER 1: ML/Deep Learning Researcher
**Summary:** The paper presents a lightweight MobileViT_v2 based model for road extraction, supervised with clDice and employing empirical gap bridging algorithms.
**Strengths:**
- Successfully compresses parameters by ~95% using a ViT backbone.
- End-to-end differentiable topology loss is well integrated.
**Major weaknesses:**
- The architectural contributions (Strip Convolutions, Channel Shift) are heavily inspired by HPLNet, making the core network less novel.
- Gap bridging via explicit geometry (tangent projection, distance thresholding) is highly heuristic. Modern ML approaches would use a Graph Neural Network (GNN) to learn this.
**Minor weaknesses:**
- Only evaluated on DeepGlobe. Needs a secondary dataset (e.g. SpaceNet or Massachusetts Roads) to prove generalizability.
**Questions to authors:**
- How does the model perform without the heuristic post-processing? Provide an ablation.
**Required changes:**
- Explicitly benchmark the contribution of the post-processing gap bridging vs the raw neural network output.

## REVIEWER 2: Remote Sensing / Computer Vision Researcher
**Summary:** The authors propose a method for extracting rural dirt roads under tree canopies and analyzing network resilience using graph centrality.
**Strengths:**
- The PMGSY motivation is extremely compelling.
- Betweenness Centrality and Resilience Index are great additions beyond standard pixel metrics.
**Major weaknesses:**
- The paper motivates the problem using "rural dirt roads" but tests on DeepGlobe, which is a mix of urban, suburban, and rural. There is no proof that the subset used exclusively represents unpaved rural roads.
- The CanopyShadowDropout augmentation simulates shadows, but real canopy occlusion also hides the road entirely. Is the model hallucinating the road underneath, or just guessing?
**Minor weaknesses:**
- The claim of "0.5m GSD" is standard for DeepGlobe, but rural Indian imagery (like Cartosat) might have different characteristics.
**Questions to authors:**
- Have you tested this on actual PMGSY satellite data?
**Required changes:**
- Clarify the nature of the DeepGlobe subset used. Acknowledge the limitation of not using native PMGSY imagery in the evaluation.

## REVIEWER 3: Skeptical Reproducibility-Focused Reviewer
**Summary:** Evaluates an edge-deployable model claiming 7.23 MB ONNX footprint and 1.6M parameters.
**Strengths:**
- The parameter counts and latency numbers are believable for a width_mult=1.0 MobileViT.
- The repository provides ONNX export scripts.
**Major weaknesses:**
- The dataset split claims "no resizing leakage", but does the codebase actually enforce random seeds for the split so it's identical across runs?
- U-Net and DeepLabv3+ baselines are sourced from literature, not reproduced in the identical framework/split. This makes the comparison unfair, as the literature models might have been evaluated on a different 10% test split.
**Minor weaknesses:**
- The alpha decay parameters ($E_{decay}=40$, $p=0.5$) seem hyper-tuned for 50 epochs.
**Questions to authors:**
- Can you provide the exact train/val/test split indices to ensure fair comparison?
**Required changes:**
- The authors must either implement and run the U-Net baseline on their exact split or explicitly add a disclaimer in the tables that baselines are from external literature and may not be perfectly comparable.

---

## Pre-submission Blockers (Action Required)
1. **Baseline Fairness:** The paper compares MobileViT (evaluated on the custom split) against literature baselines. We must add a clear disclaimer to the table/text or run a U-Net baseline locally. (Action: Add disclaimer, mostly done but make sure it's prominent).
2. **Post-processing Ablation:** The ablation study (Table 4) must clearly separate the impact of the heuristic gap-bridging from the neural network's raw topological output.
3. **Dataset Generality:** Add a note acknowledging that DeepGlobe is a proxy for PMGSY data.
