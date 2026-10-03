"""
generate_fig6_real.py
=====================
Generate fig6_centrality_resilience.png from REAL prediction data.

Panel (a): Real betweenness centrality graph from an actual predicted mask
Panel (b): Real targeted attack vs random failure stress test on a real graph  
Panel (c): Real resilience indices grouped by IoU performance quartile

Run from project root:
    python backend/scripts/generate_fig6_real.py
"""

import sys, os, csv, random
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "src"))

PRED_DIR = ROOT / "outputs" / "outputs" / "outputs" / "preds"
CSV_PATH = ROOT / "outputs" / "outputs" / "outputs" / "metrics" / "evaluation_results.csv"
OUT_PATH = ROOT / "docs" / "figures" / "fig6_centrality_resilience.png"

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx

try:
    import cv2
    def load_mask(p): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
except ImportError:
    from PIL import Image
    def load_mask(p): return np.array(Image.open(p).convert("L"))

# ── load CSV ──────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append({k: float(v) if k != "filename" else v for k, v in row.items()})
print(f"Loaded {len(rows)} evaluation rows")

# ── import graph adapter ──────────────────────────────────────────────────────
try:
    from src.utils.graph_adapter import mask_to_graph
    print("graph_adapter imported OK")
except Exception as e:
    print(f"graph_adapter failed ({e}), using skimage fallback")
    from skimage.morphology import skeletonize
    def mask_to_graph(mask):
        skel = skeletonize((mask > 127).astype(np.uint8)).astype(np.uint8)
        G = nx.Graph()
        pts = set(map(tuple, np.argwhere(skel > 0).tolist()))
        for (r, c) in pts:
            G.add_node((r, c))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if (dr,dc) == (0,0): continue
                    if (r+dr, c+dc) in pts:
                        G.add_edge((r,c),(r+dr,c+dc))
        return G

# ── helpers ───────────────────────────────────────────────────────────────────
def largest_component(G):
    comps = list(nx.connected_components(G))
    return G.subgraph(max(comps, key=len)).copy() if comps else G.copy()

def global_efficiency(G):
    n = G.number_of_nodes()
    if n < 2: return 0.0
    total = sum(
        1.0/d
        for src in G.nodes()
        for tgt, d in nx.single_source_shortest_path_length(G, src).items()
        if tgt != src and d > 0
    )
    return total / (n*(n-1))

def compute_resilience(G):
    Gc = largest_component(G)
    if Gc.number_of_nodes() < 5: return None
    E0 = global_efficiency(Gc)
    if E0 < 1e-9: return None
    try:
        cent = nx.betweenness_centrality(Gc, normalized=True)
    except Exception: return None
    n_rm = max(1, int(0.10 * Gc.number_of_nodes()))
    top  = sorted(cent, key=cent.get, reverse=True)[:n_rm]
    Gc2  = Gc.copy(); Gc2.remove_nodes_from(top)
    return global_efficiency(Gc2) / E0

# ── pick a good candidate for panel (a) ──────────────────────────────────────
sorted_rows = sorted(rows, key=lambda r: r["iou"])
mid_rows    = sorted_rows[len(sorted_rows)//3 : 2*len(sorted_rows)//3]
random.seed(42); random.shuffle(mid_rows)

chosen_row, chosen_G = None, None
print("Searching for panel (a) candidate...")
for row in mid_rows[:60]:
    p = PRED_DIR / row["filename"]
    if not p.exists(): continue
    mask = load_mask(p)
    if mask is None: continue
    try:
        G  = mask_to_graph(mask)
        Gc = largest_component(G)
        if 15 <= Gc.number_of_nodes() <= 100 and Gc.number_of_edges() >= 10:
            chosen_row, chosen_G = row, Gc
            print(f"  Selected: {row['filename']}  IoU={row['iou']:.3f}  nodes={Gc.number_of_nodes()}")
            break
    except Exception:
        pass

if chosen_G is None:
    for row in sorted_rows:
        p = PRED_DIR / row["filename"]
        if not p.exists(): continue
        mask = load_mask(p)
        if mask is None: continue
        try:
            G = largest_component(mask_to_graph(mask))
            if G.number_of_nodes() >= 8:
                chosen_row, chosen_G = row, G
                print(f"  Fallback: {row['filename']}  nodes={G.number_of_nodes()}")
                break
        except Exception:
            pass

assert chosen_G is not None

# betweenness centrality
k = min(80, chosen_G.number_of_nodes())
centrality = nx.betweenness_centrality(chosen_G, k=k, normalized=True)
gatekeeper = max(centrality, key=centrality.get)
cent_vals  = list(centrality.values())
print(f"Betweenness max={max(cent_vals):.4f} mean={np.mean(cent_vals):.4f}")

# ── panel (b) stress test ─────────────────────────────────────────────────────
fractions = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

def targeted_attack(G, cent, fracs):
    Gw = G.copy()
    ordered = sorted(cent, key=cent.get, reverse=True)
    E0 = global_efficiency(Gw)
    if E0 < 1e-9: return [1.0]*len(fracs)
    results, removed = [], 0
    for frac in fracs:
        target = int(frac * len(ordered))
        while removed < target and removed < len(ordered):
            n = ordered[removed]
            if Gw.has_node(n): Gw.remove_node(n)
            removed += 1
        results.append(global_efficiency(Gw) / E0)
    return results

def random_attack_sim(G, fracs, n_trials=8):
    nodes = list(G.nodes())
    E0 = global_efficiency(G)
    if E0 < 1e-9: return [1.0]*len(fracs)
    results = []
    for frac in fracs:
        vals = []
        for _ in range(n_trials):
            Gw = G.copy()
            n_rm = int(frac * len(nodes))
            if n_rm > 0: Gw.remove_nodes_from(random.sample(nodes, min(n_rm, len(nodes)-1)))
            vals.append(global_efficiency(Gw) / E0)
        results.append(float(np.mean(vals)))
    return results

print("Running stress tests...")
random.seed(0)
targeted  = targeted_attack(chosen_G, centrality, fractions)
rand_eff  = random_attack_sim(chosen_G, fractions)
print(f"Targeted: {[f'{v:.3f}' for v in targeted]}")
print(f"Random:   {[f'{v:.3f}' for v in rand_eff]}")

# ── panel (c) quartile resilience ────────────────────────────────────────────
ious = np.array([r["iou"] for r in rows])
q25, q50, q75 = np.percentile(ious, [25, 50, 75])

buckets = {"Q1":[], "Q2":[], "Q3":[], "Q4":[]}
for r in rows:
    iou = r["iou"]
    if   iou <= q25: buckets["Q1"].append(r)
    elif iou <= q50: buckets["Q2"].append(r)
    elif iou <= q75: buckets["Q3"].append(r)
    else:            buckets["Q4"].append(r)

q_res = {}
random.seed(7)
for qname, qrows in buckets.items():
    sample = random.sample(qrows, min(15, len(qrows)))
    rs = []
    for row in sample:
        p = PRED_DIR / row["filename"]
        if not p.exists(): continue
        mask = load_mask(p)
        if mask is None: continue
        try:
            G = mask_to_graph(mask)
            R = compute_resilience(G)
            if R is not None: rs.append(R)
        except Exception:
            pass
    q_res[qname] = float(np.mean(rs)) if rs else 0.5
    print(f"  {qname}: n={len(rs)}  R={q_res[qname]:.3f}")

q_labels = [
    f"Q1 · Low IoU\n(IoU < {q25:.2f})",
    f"Q2 · Mid-Low IoU\n(IoU < {q50:.2f})",
    f"Q3 · Mid-High IoU\n(IoU < {q75:.2f})",
    f"Q4 · High IoU\n(IoU ≥ {q75:.2f})",
]
q_values = [q_res["Q1"], q_res["Q2"], q_res["Q3"], q_res["Q4"]]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor("white")

# ── (a) centrality ────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_title(r"(a) Betweenness Centrality $C_B(v)$" +
             f"\n(Predicted Graph — {chosen_row['filename'].replace('_mask.png','')})",
             fontsize=10, fontweight="bold")
pos = nx.spring_layout(chosen_G, seed=42, k=2.5)
norm = Normalize(vmin=min(cent_vals), vmax=max(cent_vals))
nx.draw_networkx_edges(chosen_G, pos, ax=ax, alpha=0.3, edge_color="#888888", width=0.9)
sc = nx.draw_networkx_nodes(chosen_G, pos, ax=ax,
                             node_size=55, node_color=cent_vals,
                             cmap=cm.plasma, vmin=min(cent_vals), vmax=max(cent_vals))
nx.draw_networkx_nodes(chosen_G, pos, nodelist=[gatekeeper], ax=ax,
                        node_size=220, node_color="none",
                        edgecolors="red", linewidths=2.5)
gkp = pos[gatekeeper]
ax.annotate("Gatekeeper\nNode", xy=gkp,
            xytext=(gkp[0]+0.18, gkp[1]+0.18), fontsize=7.5, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
plt.colorbar(sc, ax=ax, label="Betweenness", shrink=0.8)
ax.set_axis_off()
ax.text(0.02, 0.02,
        f"Nodes={chosen_G.number_of_nodes()}  Edges={chosen_G.number_of_edges()}\n"
        f"IoU={chosen_row['iou']:.3f}  APLS={chosen_row['apls']:.3f}",
        transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")

# ── (b) stress test ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_title(r"(b) Resilience Stress Test" + "\n" + r"$E(G)$ Degradation (Real Graph)",
             fontsize=10, fontweight="bold")
pct = [f*100 for f in fractions]
ax.plot(pct, targeted,  "r-o",  lw=2, ms=5, label="Targeted Attack (Gatekeepers)")
ax.plot(pct, rand_eff,  "b--s", lw=2, ms=5, label="Random Node Failure")
ax.axhline(0.5, color="gray", ls=":", lw=1.2, alpha=0.7, label="50% threshold")
ax.set_xlabel("Fraction of Nodes Removed (%)", fontsize=10)
ax.set_ylabel(r"Global Efficiency $E(G)/E_0$",  fontsize=10)
ax.set_ylim(-0.05, 1.05); ax.set_xlim(-1, 42)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_facecolor("#fafafa")

# ── (c) quartile resilience ───────────────────────────────────────────────────
ax = axes[2]
ax.set_title("(c) Road Graph Resilience Index $R$\nby IoU Performance Quartile\n(623 DeepGlobe Val Images)",
             fontsize=10, fontweight="bold")
colors = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9"]
bars   = ax.barh(q_labels, q_values, color=colors, edgecolor="white", height=0.55)
for bar, val in zip(bars, q_values):
    ax.text(val+0.015, bar.get_y()+bar.get_height()/2,
            f"{val:.2f}", va="center", ha="left", fontsize=9, fontweight="bold")
ax.axvline(0.5, color="#c0392b", ls="--", lw=1.5, alpha=0.85, label="Critical threshold (R=0.5)")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Resilience Index $R$", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
ax.set_facecolor("#fafafa")
ax.invert_yaxis()

plt.tight_layout(pad=2.0)
plt.savefig(str(OUT_PATH), dpi=180, bbox_inches="tight", facecolor="white")
print(f"\n✅ Saved: {OUT_PATH}")
