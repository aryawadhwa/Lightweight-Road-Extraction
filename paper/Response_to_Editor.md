Original Manuscript ID: Access-2026-57395	
Original Article Title: “MobileViT Based Lightweight Road Extraction with Connectivity Aware Training for Rural Satellite Imagery”
To: IEEE Access Editor
Re: Response to Editor

Dear Editor,

Thank you for allowing a resubmission of our manuscript, with an opportunity to address your comments and concerns.

We are uploading (a) our point-by-point response to your comments (below), (b) an updated manuscript with yellow highlighting indicating changes (under “Highlighted PDF”), and (c) a clean updated manuscript without highlights (“Main Manuscript”).

Best regards,
Shanta Rangaswamy et al.

---

### Concern # 1 (please list here): 
Table 2 gives accuracy scores from other papers. However, these methods used different training and test images.

**Author response:**  
We agree completely with the Editor. Comparing accuracy metrics across disparate dataset splits (as was done in the previous Table 2) introduces confounding variables that compromise scientific rigor. Evaluating models on different training and test images does not provide an apples-to-apples basis for comparison. To eliminate this discrepancy, the previous Table 2 has been removed from the manuscript. Instead, architectural paradigms are discussed qualitatively in Section II, and all empirical accuracy comparisons have been consolidated into a unified benchmark under identical dataset conditions in Section V.

**Author action:** We updated the manuscript by removing the previous Table 2 (the cross-study literature comparison). In Section II-E (“Reported Results”), we revised the text to explain that fair comparisons require evaluating baselines under identical training and testing partitions. All quantitative accuracy comparisons are now presented strictly in the new unified benchmark table (Table V) in Section V.

---

### Concern # 2 (please list here): 
In addition, Table 5 compares model size and speed but does not compare accuracy.

**Author response:**  
We thank the Editor for this critical observation. In the original manuscript, Table 5 evaluated computational complexity (parameters, FLOPs, model size, and CPU latency) on laptop hardware, but omitted segmentation accuracy. Reporting computational efficiency in isolation fails to capture whether lightweight networks sacrifice vital topological connectivity and spatial precision. We have resolved this limitation by benchmarking the accuracy of the reference models alongside their computational cost.

**Author action:** We updated the manuscript by expanding Table 5 into a comprehensive, multi-metric benchmark table (now Table V) that reports both segmentation accuracy (Road IoU, Relaxed F1, and Centerline Dice) and computational cost (Parameters, FLOPs, Model Size, and CPU Latency) for all evaluated architectures. Sections V-A and VI-A have been rewritten to analyze the multi-dimensional trade-offs between model efficiency and topological accuracy.

---

### Concern # 3 (please list here): 
Please compare the proposed model with existing methods using the same training and test images and report both accuracy and cost.

**Author response:**  
We have fully addressed this requirement by benchmarking the proposed lightweight network alongside established segmentation architectures (U-Net, D-LinkNet34, DeepLabV3+ MobileNetV3, and LR-ASPP MobileNetV3) on the exact same 6,226-tile DeepGlobe dataset partition (5,323 training, 280 selection, and 623 held-out test tiles) evaluated at full native 1024×1024 resolution. The empirical results confirm that:
1. Compared to the heavy U-Net baseline (31.04M parameters, 1541.4 GFLOPs), our proposed network achieves 5.6 to 6.9 points higher IoU (59.8% – 61.1% vs. 54.2%) and 22.7 points higher centerline Dice (84.5% vs. 61.8%), while reducing parameter count by 94.8% and executing 7.8× to 11.6× faster on CPU.
2. Compared to the DeepGlobe challenge winner D-LinkNet34 (31.10M parameters, 212.6 GFLOPs), our network achieves higher IoU (61.1% vs. 58.1%) and substantially better topological connectivity (84.5% vs. 66.2% clDice) while requiring 19.4× fewer parameters and 3.5× fewer FLOPs.
3. Compared to fast lightweight baselines (LR-ASPP MobileNetV3, 3.22M parameters), our model improves IoU by 9.7 points (61.1% vs. 51.4%) and clDice by 25.5 points (84.5% vs. 59.0%), demonstrating that lightweight CNNs lack the global receptive field needed to traverse canopy occlusions without our transformer and strip convolution priors.

**Author action:** We updated the manuscript by:
a) Incorporating the consolidated benchmark table (Table V) in Section V-A, reporting Parameters (M), GFLOPs, Weight Size (MB), CPU Latency (s), Road IoU (%), Relaxed F1 (%), and Centerline Dice (clDice, %).
b) Revising Section I (Introduction), Section V (Results), and Section VI (Discussion) to detail the empirical findings of this identical-split benchmark.
c) Removing the limitation item in Section VI-C that previously stated “No baselines on the same split”, and replacing it with an in-depth analysis of the observed accuracy-efficiency Pareto frontier.
d) Highlighting all added and modified text and table entries in yellow in the resubmitted “Highlighted PDF”.
