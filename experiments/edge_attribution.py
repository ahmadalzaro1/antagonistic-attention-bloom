"""
Experiment Two -- Edge Attribution Patching (EAP) with Directional Attribution

Uses correct gradient-based attribution (Nanda's definition) to discover which
head-to-head edges carry the most information for each (language, category).

CRITICAL FIX: Previous version used ||act|| * ||grad|| (magnitude sensitivity),
which loses all directional information. Correct EAP uses:
    edge_score = ((clean_act - corrupted_act) * grad).sum()
This preserves sign (positive = helpful, negative = harmful) and direction.

Requires clean/corrupted prompt pairs (shares patching_pairs.json with
activation patching).

Usage:
    python exp2/edge_attribution.py exp2/prompts/patching_pairs.json
    python exp2/edge_attribution.py exp2/prompts/patching_pairs.json --top-heads 50
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("WARNING: networkx not installed. Circuit graphs will be skipped.")

from transformer_lens import HookedTransformer


LANGUAGES = ["arabic", "english", "french", "chinese", "hindi", "japanese"]
LANG_SHORT = {
    "arabic": "ar", "english": "en", "french": "fr",
    "chinese": "zh", "hindi": "hi", "japanese": "ja",
}
CATEGORIES = ["geo", "sci", "hist", "gen", "numseq", "tmpseq", "ling", "cult", "rel"]
LANG_COLORS = {
    "arabic": "#2ecc71", "english": "#e74c3c", "french": "#3498db",
    "chinese": "#f39c12", "hindi": "#9b59b6", "japanese": "#e67e22",
}


def load_pairs(path: str) -> list[dict]:
    """Load clean/corrupted prompt pairs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif "pairs" in data:
        return data["pairs"]
    raise ValueError(f"Unknown format in {path}")


def get_top_heads_from_ablation(ablation_path: str, top_n: int = 50) -> set[tuple[int, int]]:
    """Get top-N most important heads from ablation results (global across all langs)."""
    with open(ablation_path, encoding="utf-8") as f:
        data = json.load(f)

    n_layers = data["metadata"]["n_layers"]
    n_heads = data["metadata"]["n_heads"]

    grid = np.zeros((n_layers, n_heads))
    counts = np.zeros((n_layers, n_heads))
    for r in data["results"]:
        grid[r["layer"], r["head"]] += r["prob_drop"]
        counts[r["layer"], r["head"]] += 1
    grid /= np.maximum(counts, 1)

    flat = grid.flatten()
    top_idx = np.argsort(flat)[::-1][:top_n]
    return {(idx // n_heads, idx % n_heads) for idx in top_idx}


def logit_diff_metric(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    """Compute logit difference: logit[target] - max(logit[other]).

    This is the standard metric for EAP -- linear in residual stream,
    avoids softmax saturation issues.
    """
    target_logit = logits[0, -1, target_id]
    # Get max logit excluding target
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[target_id] = False
    max_other = logits[0, -1, mask].max()
    return target_logit - max_other


def compute_edge_attributions(model: HookedTransformer,
                              clean_prompt: str, corrupted_prompt: str,
                              target_id: int, top_heads: set[tuple[int, int]],
                              n_layers: int, n_heads: int) -> list[dict]:
    """Compute directional edge attribution scores using correct EAP formula.

    For each pair (src_head, dst_head) where both are in top_heads and src is
    in an earlier layer than dst:
        edge_score = ((clean_act(src) - corrupted_act(src)) * grad(dst)).sum()

    Uses TransformerLens model.add_hook() API with "bwd" direction for clean
    gradient computation. This is more memory-efficient than register_forward_hook
    + retain_grad().
    """
    model.reset_hooks()

    # Phase 1: Clean forward pass — cache activations
    clean_cache = {}

    def clean_fwd_hook(act, hook):
        clean_cache[hook.name] = act.detach().clone()

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        model.add_hook(hook_name, clean_fwd_hook, dir="fwd")

    with torch.no_grad():
        model(clean_prompt)
    model.reset_hooks()

    # Phase 2: Corrupted forward + backward — cache corrupted activations and gradients
    corrupted_cache = {}
    grad_cache = {}

    def corrupt_fwd_hook(act, hook):
        corrupted_cache[hook.name] = act.detach().clone()
        return act

    def bwd_hook(grad, hook):
        grad_cache[hook.name] = grad.detach().clone()

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        model.add_hook(hook_name, corrupt_fwd_hook, dir="fwd")
        model.add_hook(hook_name, bwd_hook, dir="bwd")

    # Forward pass with gradients enabled
    corrupted_logits = model(corrupted_prompt)
    metric = logit_diff_metric(corrupted_logits, target_id)
    metric.backward()
    model.reset_hooks()

    # Phase 3: Compute directional edge scores
    edges = []
    for (src_l, src_h) in top_heads:
        src_hook = f"blocks.{src_l}.attn.hook_z"
        if src_hook not in clean_cache or src_hook not in corrupted_cache:
            continue

        clean_act = clean_cache[src_hook][0, -1, src_h, :]
        corrupt_act = corrupted_cache[src_hook][0, -1, src_h, :]
        act_diff = clean_act - corrupt_act

        for (dst_l, dst_h) in top_heads:
            if dst_l <= src_l:
                continue  # Only forward edges

            dst_hook = f"blocks.{dst_l}.attn.hook_z"
            if dst_hook not in grad_cache:
                continue

            grad = grad_cache[dst_hook][0, -1, dst_h, :]

            # Correct directional EAP formula
            edge_score = (act_diff * grad).sum().item()

            if abs(edge_score) > 1e-8:
                edges.append({
                    "source": f"L{src_l}H{src_h}",
                    "dest": f"L{dst_l}H{dst_h}",
                    "source_layer": src_l,
                    "source_head": src_h,
                    "dest_layer": dst_l,
                    "dest_head": dst_h,
                    "attribution": round(edge_score, 8),
                })

    # Clean up
    clean_cache.clear()
    corrupted_cache.clear()
    grad_cache.clear()

    return edges


def run_eap(model: HookedTransformer, pairs: list[dict],
            top_heads: set[tuple[int, int]], n_layers: int, n_heads: int,
            pairs_per_cell: int = 3) -> dict[str, list[dict]]:
    """Run EAP on clean/corrupted pairs, grouped by (language, category)."""
    tokenizer = model.tokenizer

    # Group pairs by (language, category)
    cells: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        key = f"{p.get('group', p.get('language', 'unknown'))}_{p.get('category', 'unknown')}"
        cells[key].append(p)

    all_circuits = {}

    for cell_key, cell_pairs in sorted(cells.items()):
        subset = cell_pairs[:pairs_per_cell]
        print(f"\n  {cell_key} ({len(subset)} pairs):")

        cell_edges = []
        for p in subset:
            tids = tokenizer.encode(p["target_token"], add_special_tokens=False)
            if not tids:
                continue
            target_id = tids[0]

            clean_prompt = p["clean_prompt"]
            corrupted_prompt = p["corrupted_prompt"]

            torch.set_grad_enabled(True)
            try:
                edges = compute_edge_attributions(
                    model, clean_prompt, corrupted_prompt, target_id,
                    top_heads, n_layers, n_heads
                )
                cell_edges.extend(edges)
                n_pos = sum(1 for e in edges if e["attribution"] > 0)
                n_neg = sum(1 for e in edges if e["attribution"] < 0)
                print(f"    {p['id']}: {len(edges)} edges ({n_pos} pos, {n_neg} neg)")
            except Exception as e:
                print(f"    {p['id']}: ERROR - {e}")
            finally:
                torch.set_grad_enabled(False)
                model.zero_grad(set_to_none=True)

        if not cell_edges:
            continue

        # Aggregate: mean attribution per edge across pairs in this cell
        edge_sums: dict[tuple, list[float]] = defaultdict(list)
        for e in cell_edges:
            key = (e["source"], e["dest"])
            edge_sums[key].append(e["attribution"])

        aggregated = []
        for (src, dst), scores in edge_sums.items():
            aggregated.append({
                "source": src,
                "dest": dst,
                "mean_attribution": round(np.mean(scores), 8),
                "std_attribution": round(np.std(scores), 8) if len(scores) > 1 else 0.0,
                "n_pairs": len(scores),
            })
        # Sort by absolute value (strongest edges first, regardless of sign)
        aggregated.sort(key=lambda x: -abs(x["mean_attribution"]))

        all_circuits[cell_key] = aggregated
        top3_str = ", ".join(
            f"{e['source']}->{e['dest']}({e['mean_attribution']:+.4f})"
            for e in aggregated[:3]
        )
        print(f"    Aggregated: {len(aggregated)} unique edges, top3: {top3_str}")

    return all_circuits


# -- Visualization -----------------------------------------------------------

def plot_circuit_graph(edges: list[dict], title: str, out_path: Path,
                       top_n: int = 20):
    """Plot circuit graph using networkx. Supports positive and negative edges."""
    if not HAS_NETWORKX or not edges:
        return

    G = nx.DiGraph()
    top_edges = edges[:top_n]

    for e in top_edges:
        G.add_edge(e["source"], e["dest"], weight=e["mean_attribution"])

    if len(G.nodes()) == 0:
        return

    pos = {}
    for node in G.nodes():
        parts = node.replace("L", "").replace("H", " ").split()
        layer = int(parts[0])
        head = int(parts[1])
        pos[node] = (head, -layer)

    fig, ax = plt.subplots(figsize=(12, 10))
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    abs_weights = [abs(w) for w in weights]
    max_w = max(abs_weights) if abs_weights else 1

    # Color by sign: red=positive (helpful), blue=negative (harmful)
    edge_colors = ["#d32f2f" if w > 0 else "#1565c0" for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400, node_color="lightblue",
                           edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[2 * w / max_w + 0.5 for w in abs_weights],
                           edge_color=edge_colors,
                           alpha=0.8, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer (top = early)")

    # Legend for edge colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="Positive (helpful)"),
        Patch(facecolor="#1565c0", label="Negative (harmful)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_comparison(circuits: dict[str, list[dict]], out_dir: Path):
    """Compare circuit similarity across languages using Jaccard index."""
    lang_edges: dict[str, set] = defaultdict(set)
    for cell_key, edges in circuits.items():
        lang = cell_key.split("_")[0]
        lang_full = None
        for full, short in LANG_SHORT.items():
            if short == lang:
                lang_full = full
                break
        if lang_full is None:
            lang_full = lang

        for e in edges[:20]:
            lang_edges[lang_full].add((e["source"], e["dest"]))

    langs = [l for l in LANGUAGES if l in lang_edges]
    if len(langs) < 2:
        return

    n = len(langs)
    jaccard = np.zeros((n, n))
    for i, a in enumerate(langs):
        for j, b in enumerate(langs):
            if i == j:
                jaccard[i, j] = 1.0
            else:
                inter = len(lang_edges[a] & lang_edges[b])
                union = len(lang_edges[a] | lang_edges[b])
                jaccard[i, j] = inter / union if union > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l.capitalize() for l in langs], rotation=45)
    ax.set_yticklabels([l.capitalize() for l in langs])

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if jaccard[i, j] > 0.5 else "black")

    ax.set_title("Circuit Similarity (Jaccard Index of Top Edges)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Index")
    plt.tight_layout()
    fig.savefig(out_dir / "eap_circuit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eap_circuit_comparison.png")


def plot_edge_sign_distribution(circuits: dict[str, list[dict]], out_dir: Path):
    """Plot distribution of positive vs negative edge attributions per language."""
    lang_pos: dict[str, int] = defaultdict(int)
    lang_neg: dict[str, int] = defaultdict(int)

    for cell_key, edges in circuits.items():
        lang = cell_key.split("_")[0]
        lang_full = None
        for full, short in LANG_SHORT.items():
            if short == lang:
                lang_full = full
                break
        if lang_full is None:
            lang_full = lang

        for e in edges:
            if e["mean_attribution"] > 0:
                lang_pos[lang_full] += 1
            else:
                lang_neg[lang_full] += 1

    langs = [l for l in LANGUAGES if l in lang_pos or l in lang_neg]
    if not langs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(langs))
    width = 0.35

    pos_counts = [lang_pos.get(l, 0) for l in langs]
    neg_counts = [lang_neg.get(l, 0) for l in langs]

    ax.bar(x - width/2, pos_counts, width, label="Positive (helpful)",
           color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, neg_counts, width, label="Negative (harmful)",
           color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Language")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Directional Edge Attribution: Positive vs Negative")
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in langs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "eap_edge_signs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved eap_edge_signs.png")


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Directional Edge Attribution Patching")
    parser.add_argument("input", help="Patching pairs JSON or ablation results JSON")
    parser.add_argument("--ablation", default=None,
                        help="Ablation results JSON (for top head selection)")
    parser.add_argument("--output", default="results/exp2/eap",
                        help="Output directory")
    parser.add_argument("--top-heads", type=int, default=50,
                        help="Number of top heads to include in edge search")
    parser.add_argument("--pairs-per-cell", type=int, default=3,
                        help="Number of pairs per (language, category) cell")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    pairs = load_pairs(args.input)
    print(f"Loaded {len(pairs)} clean/corrupted pairs")

    # Get top heads from ablation if provided
    print(f"\nIdentifying top-{args.top_heads} heads...")
    if args.ablation and Path(args.ablation).exists():
        try:
            top_heads = get_top_heads_from_ablation(args.ablation, args.top_heads)
            print(f"  {len(top_heads)} top heads from ablation results")
        except (KeyError, TypeError):
            print("  Could not extract top heads. Using all heads in layers 0-10.")
            top_heads = {(l, h) for l in range(10) for h in range(16)}
    else:
        print("  No ablation data. Using all heads in layers 0-10.")
        top_heads = {(l, h) for l in range(10) for h in range(16)}

    # Load model
    print("\nLoading BLOOM-560m on CPU...")
    model = HookedTransformer.from_pretrained("bigscience/bloom-560m", device="cpu")
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Run EAP
    print(f"\nRunning Directional Edge Attribution Patching "
          f"({len(top_heads)} heads, {args.pairs_per_cell} pairs/cell)...")
    t0 = time.time()
    circuits = run_eap(model, pairs, top_heads, n_layers, n_heads,
                       args.pairs_per_cell)
    elapsed = time.time() - t0
    print(f"\n  EAP complete in {elapsed:.1f}s")

    # Save results
    output_data = {
        "metadata": {
            "model": "bigscience/bloom-560m",
            "technique": "directional_edge_attribution_patching",
            "formula": "((clean_act - corrupted_act) * grad).sum()",
            "metric": "logit_difference",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "top_heads_count": len(top_heads),
            "pairs_per_cell": args.pairs_per_cell,
            "top_heads": [f"L{l}H{h}" for l, h in sorted(top_heads)],
            "note": ("Directional EAP preserves sign: positive edges are helpful "
                     "(carry information toward correct prediction), negative edges "
                     "are harmful. Previous magnitude-based formula (||act||*||grad||) "
                     "lost this information."),
        },
        "circuits": circuits,
    }
    results_path = out_dir / "eap_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("EAP SUMMARY (Directional Attribution)")
    print(f"{'=' * 60}")
    for cell_key, edges in sorted(circuits.items()):
        n_edges = len(edges)
        n_pos = sum(1 for e in edges if e["mean_attribution"] > 0)
        n_neg = n_edges - n_pos
        top_edge = edges[0] if edges else None
        top_str = (f"{top_edge['source']}->{top_edge['dest']} "
                   f"({top_edge['mean_attribution']:+.4f})" if top_edge else "none")
        print(f"  {cell_key:20s}: {n_edges} edges ({n_pos}+/{n_neg}-), top: {top_str}")

    # Visualizations
    print("\nGenerating visualizations...")

    if HAS_NETWORKX:
        lang_circuits: dict[str, list[dict]] = defaultdict(list)
        for cell_key, edges in circuits.items():
            lang = cell_key.split("_")[0]
            lang_circuits[lang].extend(edges)

        for lang, edges in lang_circuits.items():
            edge_sums: dict[tuple, list[float]] = defaultdict(list)
            for e in edges:
                edge_sums[(e["source"], e["dest"])].append(e["mean_attribution"])
            agg = [{"source": s, "dest": d, "mean_attribution": np.mean(v)}
                   for (s, d), v in edge_sums.items()]
            agg.sort(key=lambda x: -abs(x["mean_attribution"]))

            lang_full = None
            for full, short in LANG_SHORT.items():
                if short == lang:
                    lang_full = full
                    break
            title = f"EAP Circuit: {(lang_full or lang).capitalize()} (top 20 edges)"
            plot_circuit_graph(agg, title, out_dir / f"eap_circuit_{lang}.png")
            print(f"  Saved eap_circuit_{lang}.png")

    plot_circuit_comparison(circuits, out_dir)
    plot_edge_sign_distribution(circuits, out_dir)

    print(f"\n{'=' * 60}")
    print(f"EAP analysis complete! Output in {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
