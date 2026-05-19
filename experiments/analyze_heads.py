"""
Multi-Head Cross-Group Analysis

Reads ablation_results.json files and produces deeper cross-group analysis:
pairwise correlations, overlap analysis, antagonistic heads, category
breakdowns, and visualizations.

Usage:
    python analyze_heads.py results/crosslingual/crosslingual_results.json
    python analyze_heads.py results/four_languages/ablation_results.json --top-n 20
    python analyze_heads.py results/four_languages/ablation_results.json --output-dir analysis/
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_results(path: str) -> tuple[dict, list[dict], int, int]:
    """Load JSON results, normalizing 'language' -> 'group' schema."""
    with open(path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]

    # Schema compat: run_crosslingual.py uses "language", engine uses "group"
    for r in results:
        if "language" in r and "group" not in r:
            r["group"] = r["language"]
        elif "group" in r and "language" not in r:
            r["language"] = r["group"]

    n_layers = metadata["n_layers"]
    n_heads = metadata["n_heads"]

    # Detect groups from results if not in metadata
    if "groups" not in metadata:
        metadata["groups"] = sorted({r["group"] for r in results})

    return metadata, results, n_layers, n_heads


# ── Grid Building ────────────────────────────────────────────────────────────

def build_group_grids(results: list[dict], n_layers: int, n_heads: int) -> dict[str, np.ndarray]:
    """Build {group: (n_layers, n_heads) grid} of mean prob_drop."""
    groups = {}
    for r in results:
        g = r["group"]
        groups.setdefault(g, []).append(r)

    grids = {}
    for gname, rows in groups.items():
        grid = np.zeros((n_layers, n_heads))
        counts = np.zeros((n_layers, n_heads))
        for r in rows:
            grid[r["layer"], r["head"]] += r["prob_drop"]
            counts[r["layer"], r["head"]] += 1
        grids[gname] = grid / np.maximum(counts, 1)

    return grids


def build_category_grids(results: list[dict], n_layers: int, n_heads: int) -> dict[str, dict[str, np.ndarray]]:
    """Build {group: {category: grid}} for per-category analysis."""
    buckets: dict[str, dict[str, list]] = {}
    for r in results:
        g = r["group"]
        c = r.get("category", "unknown")
        buckets.setdefault(g, {}).setdefault(c, []).append(r)

    grids = {}
    for gname, cats in buckets.items():
        grids[gname] = {}
        for cname, rows in cats.items():
            grid = np.zeros((n_layers, n_heads))
            counts = np.zeros((n_layers, n_heads))
            for r in rows:
                grid[r["layer"], r["head"]] += r["prob_drop"]
                counts[r["layer"], r["head"]] += 1
            grids[gname][cname] = grid / np.maximum(counts, 1)

    return grids


# ── Analysis Functions ───────────────────────────────────────────────────────

def pairwise_correlations(grids: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Pearson r for all group pairs."""
    pairs = {}
    group_names = sorted(grids.keys())
    for g1, g2 in combinations(group_names, 2):
        r, _ = pearsonr(grids[g1].flatten(), grids[g2].flatten())
        pairs[(g1, g2)] = round(r, 4)
    return pairs


def overlap_analysis(grids: dict[str, np.ndarray], top_n: int = 20) -> dict[tuple[str, str], dict]:
    """Jaccard similarity of top-N head sets per group pair."""
    group_names = sorted(grids.keys())
    top_sets = {}
    for gname, grid in grids.items():
        flat = [(l, h, grid[l, h]) for l in range(grid.shape[0]) for h in range(grid.shape[1])]
        flat.sort(key=lambda x: x[2], reverse=True)
        top_sets[gname] = {f"L{l}H{h}" for l, h, _ in flat[:top_n]}

    overlaps = {}
    for g1, g2 in combinations(group_names, 2):
        s1, s2 = top_sets[g1], top_sets[g2]
        intersection = s1 & s2
        union = s1 | s2
        jaccard = len(intersection) / len(union) if union else 0.0
        overlaps[(g1, g2)] = {
            "jaccard": round(jaccard, 4),
            "shared_heads": sorted(intersection),
            "only_g1": sorted(s1 - s2),
            "only_g2": sorted(s2 - s1),
        }
    return overlaps


def antagonistic_heads(grids: dict[str, np.ndarray], threshold: float = 0.005) -> list[dict]:
    """Heads where ablation helps one group but hurts another."""
    group_names = sorted(grids.keys())
    heads = []
    n_layers, n_heads = next(iter(grids.values())).shape

    for l in range(n_layers):
        for h in range(n_heads):
            drops = {g: grids[g][l, h] for g in group_names}
            for g1, g2 in combinations(group_names, 2):
                # g1 positive drop (ablation hurts) and g2 negative (ablation helps)
                if drops[g1] > threshold and drops[g2] < -threshold:
                    heads.append({
                        "head": f"L{l}H{h}",
                        "layer": l,
                        "head_idx": h,
                        "helped": g2,
                        "hurt": g1,
                        f"{g1}_drop": round(drops[g1], 6),
                        f"{g2}_drop": round(drops[g2], 6),
                        "magnitude": round(drops[g1] - drops[g2], 6),
                    })
                if drops[g2] > threshold and drops[g1] < -threshold:
                    heads.append({
                        "head": f"L{l}H{h}",
                        "layer": l,
                        "head_idx": h,
                        "helped": g1,
                        "hurt": g2,
                        f"{g2}_drop": round(drops[g2], 6),
                        f"{g1}_drop": round(drops[g1], 6),
                        "magnitude": round(drops[g2] - drops[g1], 6),
                    })

    heads.sort(key=lambda x: x["magnitude"], reverse=True)
    return heads


def category_breakdown(results: list[dict], n_layers: int, n_heads: int) -> dict:
    """Per-category grids + within/across-group correlations."""
    cat_grids = build_category_grids(results, n_layers, n_heads)

    # Within-group: how correlated are different categories within the same language?
    within = {}
    for gname, cats in cat_grids.items():
        cat_names = sorted(cats.keys())
        if len(cat_names) < 2:
            continue
        for c1, c2 in combinations(cat_names, 2):
            r, _ = pearsonr(cats[c1].flatten(), cats[c2].flatten())
            within.setdefault(gname, {})[(c1, c2)] = round(r, 4)

    # Across-group: same category, different languages
    all_groups = sorted(cat_grids.keys())
    all_cats = sorted({c for cats in cat_grids.values() for c in cats})
    across = {}
    for cat in all_cats:
        groups_with_cat = [g for g in all_groups if cat in cat_grids[g]]
        if len(groups_with_cat) < 2:
            continue
        for g1, g2 in combinations(groups_with_cat, 2):
            r, _ = pearsonr(cat_grids[g1][cat].flatten(), cat_grids[g2][cat].flatten())
            across.setdefault(cat, {})[(g1, g2)] = round(r, 4)

    return {
        "grids": cat_grids,
        "within_group": within,
        "across_group": across,
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_correlation_matrix(correlations: dict[tuple[str, str], float], group_names: list[str], output_dir: Path):
    """NxN heatmap of pairwise r values."""
    n = len(group_names)
    matrix = np.eye(n)
    for (g1, g2), r in correlations.items():
        i, j = group_names.index(g1), group_names.index(g2)
        matrix[i, j] = r
        matrix[j, i] = r

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.3)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticklabels(group_names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 0.5 else "black", fontsize=11)

    ax.set_title("Pairwise Head Importance Correlation (Pearson r)")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_antagonistic_bar(antag_heads: list[dict], output_dir: Path, max_show: int = 25):
    """Bar chart of top antagonistic heads."""
    if not antag_heads:
        return

    show = antag_heads[:max_show]
    labels = [f"{h['head']} ({h['hurt']}+/{h['helped']}-)" for h in show]
    magnitudes = [h["magnitude"] for h in show]

    fig, ax = plt.subplots(figsize=(10, max(6, len(show) * 0.35)))
    colors = ["#d32f2f" if m > 0.01 else "#ff8a80" for m in magnitudes]
    ax.barh(range(len(show)), magnitudes, color=colors, alpha=0.85)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Antagonistic Magnitude (hurt_drop - helped_drop)")
    ax.set_title("Antagonistic Heads: Help One Group, Hurt Another")
    plt.tight_layout()
    fig.savefig(output_dir / "antagonistic_heads.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmaps(cat_data: dict, n_layers: int, n_heads: int, output_dir: Path):
    """Small multiples: one heatmap per (group, category)."""
    cat_grids = cat_data["grids"]
    all_groups = sorted(cat_grids.keys())
    all_cats = sorted({c for cats in cat_grids.values() for c in cats})

    if not all_groups or not all_cats:
        return

    ncols = len(all_cats)
    nrows = len(all_groups)

    # Find global vmax for consistent color scale
    vmax = 0
    for g in all_groups:
        for c in all_cats:
            if c in cat_grids[g]:
                vmax = max(vmax, abs(cat_grids[g][c]).max())
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False, sharey=True)

    for i, g in enumerate(all_groups):
        for j, c in enumerate(all_cats):
            ax = axes[i][j]
            if c in cat_grids[g]:
                im = ax.imshow(cat_grids[g][c], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_title(f"{g} / {c}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{g} / {c} (no data)", fontsize=10)
            if j == 0:
                ax.set_ylabel("Layer")
            if i == nrows - 1:
                ax.set_xlabel("Head")

    fig.suptitle("Head Importance by Group and Category", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes, label="Mean P(target) drop", shrink=0.6, pad=0.02)
    plt.tight_layout()
    fig.savefig(output_dir / "category_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_venn_text(overlaps: dict, output_dir: Path):
    """Text-based overlap summary as a figure (avoids matplotlib-venn dependency)."""
    if not overlaps:
        return

    lines = []
    for (g1, g2), data in overlaps.items():
        lines.append(f"{g1} vs {g2}:")
        lines.append(f"  Jaccard = {data['jaccard']:.3f}")
        lines.append(f"  Shared ({len(data['shared_heads'])}): {', '.join(data['shared_heads'][:10])}")
        if len(data['shared_heads']) > 10:
            lines.append(f"    ... and {len(data['shared_heads']) - 10} more")
        lines.append(f"  Only {g1} ({len(data['only_g1'])}): {', '.join(data['only_g1'][:8])}")
        lines.append(f"  Only {g2} ({len(data['only_g2'])}): {', '.join(data['only_g2'][:8])}")
        lines.append("")

    fig, ax = plt.subplots(figsize=(12, max(4, len(lines) * 0.3)))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, family="monospace", verticalalignment="top")
    ax.set_title(f"Top-N Head Overlap Analysis", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "overlap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def save_report(metadata: dict, correlations: dict, overlaps: dict,
                antag_heads: list, cat_data: dict, output_dir: Path):
    """Save JSON summary + analysis_report.md."""
    # JSON summary
    json_report = {
        "model": metadata.get("model", "unknown"),
        "groups": metadata.get("groups", []),
        "n_layers": metadata["n_layers"],
        "n_heads": metadata["n_heads"],
        "pairwise_correlations": {f"{g1}_vs_{g2}": r for (g1, g2), r in correlations.items()},
        "overlap_jaccard": {f"{g1}_vs_{g2}": data["jaccard"] for (g1, g2), data in overlaps.items()},
        "antagonistic_heads_count": len(antag_heads),
        "top_antagonistic": antag_heads[:10],
        "category_correlations": {
            "within_group": {
                g: {f"{c1}_vs_{c2}": r for (c1, c2), r in pairs.items()}
                for g, pairs in cat_data.get("within_group", {}).items()
            },
            "across_group": {
                cat: {f"{g1}_vs_{g2}": r for (g1, g2), r in pairs.items()}
                for cat, pairs in cat_data.get("across_group", {}).items()
            },
        },
    }
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # Markdown report
    groups = metadata.get("groups", [])
    lines = [
        f"# Cross-Group Head Analysis Report",
        f"",
        f"**Model:** {metadata.get('model', 'unknown')}",
        f"**Groups:** {', '.join(groups)}",
        f"**Architecture:** {metadata['n_layers']} layers x {metadata['n_heads']} heads "
        f"= {metadata['n_layers'] * metadata['n_heads']} total heads",
        f"",
        f"## Pairwise Correlations",
        f"",
        f"| Group Pair | Pearson r | Interpretation |",
        f"|------------|-----------|----------------|",
    ]
    for (g1, g2), r in sorted(correlations.items()):
        if r > 0.7:
            interp = "High — shared circuits"
        elif r > 0.3:
            interp = "Moderate — mixed"
        else:
            interp = "Low — independent circuits"
        lines.append(f"| {g1} vs {g2} | {r:.4f} | {interp} |")

    lines += [
        f"",
        f"## Top-N Head Overlap (Jaccard)",
        f"",
        f"| Group Pair | Jaccard | Shared Heads |",
        f"|------------|---------|--------------|",
    ]
    for (g1, g2), data in sorted(overlaps.items()):
        shared = ", ".join(data["shared_heads"][:5])
        if len(data["shared_heads"]) > 5:
            shared += f" (+{len(data['shared_heads']) - 5} more)"
        lines.append(f"| {g1} vs {g2} | {data['jaccard']:.3f} | {shared} |")

    if antag_heads:
        lines += [
            f"",
            f"## Antagonistic Heads ({len(antag_heads)} found)",
            f"",
            f"Heads that help one group when ablated but hurt another.",
            f"",
            f"| Head | Hurt | Helped | Magnitude |",
            f"|------|------|--------|-----------|",
        ]
        for h in antag_heads[:15]:
            lines.append(f"| {h['head']} | {h['hurt']} | {h['helped']} | {h['magnitude']:.4f} |")

    # Category correlations
    within = cat_data.get("within_group", {})
    across = cat_data.get("across_group", {})
    if within or across:
        lines += [f"", f"## Category Analysis", f""]

        if within:
            lines += [f"### Within-Group (same language, different categories)", f""]
            for g, pairs in sorted(within.items()):
                lines.append(f"**{g}:**")
                for (c1, c2), r in sorted(pairs.items()):
                    lines.append(f"- {c1} vs {c2}: r = {r:.4f}")
                lines.append("")

        if across:
            lines += [f"### Across-Group (same category, different languages)", f""]
            for cat, pairs in sorted(across.items()):
                lines.append(f"**{cat}:**")
                for (g1, g2), r in sorted(pairs.items()):
                    lines.append(f"- {g1} vs {g2}: r = {r:.4f}")
                lines.append("")

    lines += [
        f"",
        f"## Generated Files",
        f"",
        f"- `analysis_summary.json` — Machine-readable summary",
        f"- `correlation_matrix.png` — Pairwise correlation heatmap",
        f"- `antagonistic_heads.png` — Bar chart of antagonistic heads",
        f"- `category_heatmaps.png` — Per-(group, category) importance heatmaps",
        f"- `overlap_analysis.png` — Top-N head overlap between groups",
    ]

    with open(output_dir / "analysis_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Head Cross-Group Analysis")
    parser.add_argument("results_json", help="Path to ablation_results.json")
    parser.add_argument("--top-n", type=int, default=20, help="Top N heads for overlap analysis")
    parser.add_argument("--output-dir", help="Output directory (default: analysis/ next to input)")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Min prob_drop magnitude for antagonistic head detection")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}...")
    metadata, results, n_layers, n_heads = load_results(str(results_path))
    group_names = sorted(metadata["groups"])
    print(f"  Model: {metadata.get('model', 'unknown')}")
    print(f"  Groups: {group_names}")
    print(f"  Architecture: {n_layers}L x {n_heads}H = {n_layers * n_heads} heads")
    print(f"  Results: {len(results)} rows")

    # Build grids
    print("\nBuilding group grids...")
    grids = build_group_grids(results, n_layers, n_heads)

    # Pairwise correlations
    print("Computing pairwise correlations...")
    correlations = pairwise_correlations(grids)
    for (g1, g2), r in sorted(correlations.items()):
        print(f"  {g1} vs {g2}: r = {r:.4f}")

    # Overlap analysis
    print(f"\nTop-{args.top_n} head overlap analysis...")
    overlaps = overlap_analysis(grids, top_n=args.top_n)
    for (g1, g2), data in sorted(overlaps.items()):
        print(f"  {g1} vs {g2}: Jaccard = {data['jaccard']:.3f} "
              f"({len(data['shared_heads'])} shared)")

    # Antagonistic heads
    print(f"\nAntagonistic heads (threshold={args.threshold})...")
    antag = antagonistic_heads(grids, threshold=args.threshold)
    print(f"  Found {len(antag)} antagonistic heads")
    for h in antag[:5]:
        print(f"  {h['head']}: helps {h['helped']}, hurts {h['hurt']} "
              f"(magnitude={h['magnitude']:.4f})")

    # Category breakdown
    print("\nCategory breakdown...")
    cat_data = category_breakdown(results, n_layers, n_heads)

    # Plots
    print(f"\nGenerating visualizations in {output_dir}/...")
    plot_correlation_matrix(correlations, group_names, output_dir)
    print("  correlation_matrix.png")
    plot_antagonistic_bar(antag, output_dir)
    print("  antagonistic_heads.png")
    plot_category_heatmaps(cat_data, n_layers, n_heads, output_dir)
    print("  category_heatmaps.png")
    plot_overlap_venn_text(overlaps, output_dir)
    print("  overlap_analysis.png")

    # Report
    print("\nSaving report...")
    save_report(metadata, correlations, overlaps, antag, cat_data, output_dir)
    print("  analysis_summary.json")
    print("  analysis_report.md")

    print("\nDone!")


if __name__ == "__main__":
    main()
