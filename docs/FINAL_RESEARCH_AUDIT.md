# FINAL RESEARCH AUDIT

## A. Problem Formulation
- **Original State:** The paper claimed to solve rural road extraction using weak supervision, but presented hyperbolic claims about "perfect alignment."
- **Current State:** Reframed to focus on topology-preserving extraction under extreme computational constraints for rural auditing (PMGSY). 

## B. Literature & Novelty (The HPLNet Issue)
- **Original State:** Claimed "Directional Strip Convolution" and "Zero-Parameter Channel Shift" as completely novel contributions.
- **Current State [FIXED]:** Forensic audit revealed these were derived from HPLNet. The paper was rewritten to acknowledge HPLNet as the origin of these modules. The actual novelty is the integration of these lightweight modules with the clDice topology-preserving loss and graph-heuristic gap bridging on edge devices.

## C. Dataset Construction & Leakage
- **Original State [CRITICAL FLAW]:** DeepGlobe 1024x1024 images were randomly cropped during training but resized during validation. This destroyed physical scale consistency (roads became 4x thinner in validation). Furthermore, the 1243 validation/test split claims were statistically incongruent with DeepGlobe's public labels.
- **Current State [FIXED]:** The codebase (`dataset.py`) was rewritten to remove validation resizing. The paper was rewritten to state that spatial resolution (0.5m/px) is strictly maintained across the 80/10/10 image-level split.

## D. Reproducibility
- **Original State:** Placeholder GitHub links, missing environment files, untraceable ONNX compression claims.
- **Current State [FIXED]:** Placeholder links removed. Codebase patched. The entire training pipeline has been packaged and is currently executing reproducibly on Kaggle GPUs.

## E. Claims vs Evidence
- **Original State:** The paper claimed 3 random seeds were used, but the codebase only supported single runs. 
- **Current State:** The claims in the paper have been strictly aligned with what the codebase is actually capable of performing. Waiting on Kaggle for the final empirical metrics.
