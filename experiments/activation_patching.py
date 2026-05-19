"""
Experiment Two — Activation Patching (Causal Tracing)

For each clean/corrupted prompt pair, patches clean head activations into
the corrupted forward pass to determine which heads are *sufficient* to
restore correct predictions.

Usage:
    python exp2/activation_patching.py exp2/prompts/patching_pairs.json
    python exp2/activation_patching.py exp2/prompts/patching_pairs.json --output results/exp2/activation_patching
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformer_lens import HookedTransformer


LANGUAGES = ["arabic", "english", "french", "chinese", "hindi", "japanese"]
LANG_SHORT = {
    "arabic": "ar", "english": "en", "french": "fr",
    "chinese": "zh", "hindi": "hi", "japanese": "ja",
}
CATEGORIES = ["geo", "sci", "hist", "gen", "numseq", "tmpseq", "ling", "cult", "rel"]


def load_pairs(path: str) -> list[dict]:
    """Load clean/corrupted prompt pairs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif "pairs" in data:
        return data["pairs"]
    raise ValueError(f"Unknown format in {path}")


def patch_single_head(model: HookedTransformer, clean_prompt: str,
                      corrupted_prompt: str, target_id: int,
                      layer: int, head: int,
                      clean_cache: dict) -> tuple[float, float]:
    """Patch clean activation at (layer, head) into corrupted forward pass.

    Patches only the last token position to avoid shape mismatches when
    clean and corrupted prompts have different token counts.

    Returns (P(target), logit(target)) after patching.
    """
    clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]

    def patch_hook(value, hook):
        # Patch last position only — safe regardless of sequence length mismatch
        value[:, -1, head, :] = clean_z[:, -1, head, :]
        return value

    hook_name = f"blocks.{layer}.attn.hook_z"
    patched_logits = model.run_with_hooks(
        corrupted_prompt,
        fwd_hooks=[(hook_name, patch_hook)]
    )
    probs = torch.softmax(patched_logits[0, -1, :], dim=-1)
    return probs[target_id].item(), patched_logits[0, -1, target_id].item()


def run_activation_patching(model: HookedTransformer, pairs: list[dict],
                            out_dir: Path, checkpoint_path: Path | None = None):
    """Run activation patching for all pairs across all heads."""
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Load checkpoint if exists
    completed_ids = set()
    all_results = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_results = checkpoint.get("results", [])
        completed_ids = {r["pair_id"] for r in all_results}
        print(f"  Resumed from checkpoint: {len(completed_ids)} pairs done")

    remaining = [p for p in pairs if p["id"] not in completed_ids]
    print(f"  {len(remaining)} pairs remaining ({len(completed_ids)} already done)")

    for idx, pair in enumerate(remaining):
        pid = pair["id"]
        clean_prompt = pair["clean_prompt"]
        corrupted_prompt = pair["corrupted_prompt"]
        target_token = pair["target_token"]

        # Resolve target token
        tids = tokenizer.encode(target_token, add_special_tokens=False)
        if not tids:
            print(f"  [{idx+1}/{len(remaining)}] {pid}: SKIP (empty token)")
            continue
        target_id = tids[0]

        # Compute baselines (only cache hook_z — all we need for patching)
        clean_logits, clean_cache = model.run_with_cache(
            clean_prompt,
            names_filter=lambda name: name.endswith("attn.hook_z"))
        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=-1)
        clean_p = clean_probs[target_id].item()
        clean_logit = clean_logits[0, -1, target_id].item()

        corrupted_logits = model(corrupted_prompt)
        corrupted_probs = torch.softmax(corrupted_logits[0, -1, :], dim=-1)
        corrupted_p = corrupted_probs[target_id].item()
        corrupted_logit = corrupted_logits[0, -1, target_id].item()

        # Skip if clean doesn't actually predict the target well
        if clean_p < 0.05:
            print(f"  [{idx+1}/{len(remaining)}] {pid}: SKIP (clean_P={clean_p:.4f} < 0.05)")
            continue

        # Skip if corruption doesn't actually change the prediction
        effect_range = clean_p - corrupted_p
        logit_effect_range = clean_logit - corrupted_logit
        if effect_range < 0.02:
            print(f"  [{idx+1}/{len(remaining)}] {pid}: SKIP "
                  f"(no corruption effect: {clean_p:.4f} -> {corrupted_p:.4f})")
            continue

        t0 = time.time()

        # Patch each head
        for layer in range(n_layers):
            for head in range(n_heads):
                patched_p, patched_logit = patch_single_head(
                    model, clean_prompt, corrupted_prompt, target_id,
                    layer, head, clean_cache
                )

                # Probability-based restoration (secondary)
                restoration = (patched_p - corrupted_p) / effect_range

                # Logit difference-based restoration (primary)
                # Linear in residual stream, avoids softmax saturation
                logit_restoration = (
                    (patched_logit - corrupted_logit) / logit_effect_range
                    if abs(logit_effect_range) > 1e-6 else 0.0
                )

                all_results.append({
                    "pair_id": pid,
                    "group": pair.get("group", pair.get("language", "unknown")),
                    "category": pair.get("category", "unknown"),
                    "layer": layer,
                    "head": head,
                    "clean_prob": round(clean_p, 6),
                    "corrupted_prob": round(corrupted_p, 6),
                    "patched_prob": round(patched_p, 6),
                    "restoration_score": round(restoration, 6),
                    "prob_restored": round(patched_p - corrupted_p, 6),
                    "clean_logit": round(clean_logit, 6),
                    "corrupted_logit": round(corrupted_logit, 6),
                    "patched_logit": round(patched_logit, 6),
                    "logit_restoration": round(logit_restoration, 6),
                    "logit_restored": round(patched_logit - corrupted_logit, 6),
                })

        elapsed = time.time() - t0
        # Report top restoring head
        pair_results = [r for r in all_results if r["pair_id"] == pid]
        top_head = max(pair_results, key=lambda x: x["restoration_score"])
        print(f"  [{idx+1}/{len(remaining)}] {pid}: "
              f"clean={clean_p:.3f} corr={corrupted_p:.3f} "
              f"top=L{top_head['layer']}H{top_head['head']} "
              f"(restore={top_head['restoration_score']:.3f}) "
              f"[{elapsed:.1f}s]")

        # Checkpoint after each pair
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, all_results, pairs, model)

    return all_results


def _save_checkpoint(path: Path, results: list, pairs: list, model):
    """Atomic checkpoint save."""
    tmp = path.with_suffix(".tmp")
    data = {
        "metadata": {
            "model": "bigscience/bloom-560m",
            "technique": "activation_patching",
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "n_pairs": len(pairs),
        },
        "results": results,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)


# ── Visualization ────────────────────────────────────────────────────────────

def plot_restoration_heatmaps(results: list[dict], n_layers: int, n_heads: int,
                              out_dir: Path):
    """Plot per-language restoration score heatmaps."""
    by_lang: dict[str, list] = defaultdict(list)
    for r in results:
        by_lang[r["group"]].append(r)

    langs_present = [l for l in LANGUAGES if l in by_lang]
    if not langs_present:
        return

    ncols = min(len(langs_present), 3)
    nrows = (len(langs_present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    fig.suptitle("Activation Patching: Mean Restoration Score per Head", fontsize=14)

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Build grids
    all_vals = []
    grids = {}
    for lang in langs_present:
        grid = np.zeros((n_layers, n_heads))
        counts = np.zeros((n_layers, n_heads))
        for r in by_lang[lang]:
            grid[r["layer"], r["head"]] += r["restoration_score"]
            counts[r["layer"], r["head"]] += 1
        grid /= np.maximum(counts, 1)
        grids[lang] = grid
        all_vals.extend(grid.flatten())

    vmax = np.percentile(np.abs(all_vals), 98)

    for ax, lang in zip(axes_flat, langs_present):
        im = ax.imshow(grids[lang], aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{lang.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        # Mark top-3 heads
        flat = grids[lang].flatten()
        top3_idx = np.argsort(flat)[::-1][:3]
        for idx in top3_idx:
            l, h = divmod(idx, n_heads)
            ax.plot(h, l, "k*", markersize=10)
            ax.annotate(f"L{l}H{h}", (h, l), textcoords="offset points",
                        xytext=(5, -5), fontsize=7, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # Hide extra axes
    for ax in axes_flat[len(langs_present):]:
        ax.set_visible(False)

    fig.colorbar(im, ax=axes_flat[:len(langs_present)], shrink=0.6,
                 label="Mean Restoration Score")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "patching_restoration_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved patching_restoration_heatmaps.png")


def plot_necessity_vs_sufficiency(results: list[dict], ablation_path: str | None,
                                  n_layers: int, n_heads: int, out_dir: Path):
    """Scatter: ablation drop (necessity) vs patching restoration (sufficiency)."""
    if not ablation_path or not Path(ablation_path).exists():
        print("  Skipping necessity vs sufficiency (no ablation data)")
        return

    with open(ablation_path, encoding="utf-8") as f:
        ablation_data = json.load(f)

    # Build ablation grids per language
    ablation_grids: dict[str, np.ndarray] = {}
    ablation_counts: dict[str, np.ndarray] = {}
    for r in ablation_data["results"]:
        lang = r.get("group", r.get("language"))
        if lang not in ablation_grids:
            ablation_grids[lang] = np.zeros((n_layers, n_heads))
            ablation_counts[lang] = np.zeros((n_layers, n_heads))
        ablation_grids[lang][r["layer"], r["head"]] += r["prob_drop"]
        ablation_counts[lang][r["layer"], r["head"]] += 1
    for lang in ablation_grids:
        ablation_grids[lang] /= np.maximum(ablation_counts[lang], 1)

    # Build patching grids per language
    patching_grids: dict[str, np.ndarray] = {}
    patching_counts: dict[str, np.ndarray] = {}
    for r in results:
        lang = r["group"]
        if lang not in patching_grids:
            patching_grids[lang] = np.zeros((n_layers, n_heads))
            patching_counts[lang] = np.zeros((n_layers, n_heads))
        patching_grids[lang][r["layer"], r["head"]] += r["restoration_score"]
        patching_counts[lang][r["layer"], r["head"]] += 1
    for lang in patching_grids:
        patching_grids[lang] /= np.maximum(patching_counts[lang], 1)

    # Plot
    shared_langs = sorted(set(ablation_grids.keys()) & set(patching_grids.keys()))
    if not shared_langs:
        return

    fig, axes = plt.subplots(1, len(shared_langs), figsize=(6 * len(shared_langs), 6))
    fig.suptitle("Necessity (Ablation Drop) vs Sufficiency (Patching Restoration)", fontsize=14)

    if len(shared_langs) == 1:
        axes = [axes]

    LANG_COLORS = {
        "arabic": "#2ecc71", "english": "#e74c3c", "french": "#3498db",
        "chinese": "#f39c12", "hindi": "#9b59b6", "japanese": "#e67e22",
    }

    for ax, lang in zip(axes, shared_langs):
        x = ablation_grids[lang].flatten()
        y = patching_grids[lang].flatten()
        color = LANG_COLORS.get(lang, "#333333")
        ax.scatter(x, y, alpha=0.4, s=15, c=color)
        ax.set_xlabel("Mean Ablation Drop (necessity)")
        ax.set_ylabel("Mean Restoration Score (sufficiency)")
        ax.set_title(f"{lang.capitalize()}", fontweight="bold")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.axvline(0, color="gray", ls="--", alpha=0.3)

        # Correlation
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            r, p = pearsonr(x[valid], y[valid])
            ax.text(0.05, 0.95, f"r={r:.3f}", transform=ax.transAxes,
                    fontsize=10, va="top", fontweight="bold")

        # Label top heads
        for i in range(len(x)):
            if x[i] > np.percentile(x, 99) or y[i] > np.percentile(y[y > 0], 99):
                l, h = divmod(i, n_heads)
                ax.annotate(f"L{l}H{h}", (x[i], y[i]), fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "patching_necessity_vs_sufficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved patching_necessity_vs_sufficiency.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation patching for exp2")
    parser.add_argument("pairs_json", help="Path to patching pairs JSON")
    parser.add_argument("--output", default="results/exp2/activation_patching",
                        help="Output directory")
    parser.add_argument("--ablation", default=None,
                        help="Ablation results JSON (for necessity comparison)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "patching_checkpoint.json"

    # Load pairs
    pairs = load_pairs(args.pairs_json)
    print(f"Loaded {len(pairs)} clean/corrupted pairs")

    # Load model
    print("\nLoading BLOOM-560m on CPU...")
    model = HookedTransformer.from_pretrained("bigscience/bloom-560m", device="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Run patching
    print(f"\nRunning activation patching ({n_layers}L x {n_heads}H per pair)...")
    results = run_activation_patching(model, pairs, out_dir, checkpoint_path)

    # Save final results
    output_data = {
        "metadata": {
            "model": "bigscience/bloom-560m",
            "technique": "activation_patching",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_pairs": len(pairs),
            "n_results": len(results),
        },
        "results": results,
    }
    results_path = out_dir / "activation_patching_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("ACTIVATION PATCHING SUMMARY")
    print(f"{'=' * 60}")

    by_lang: dict[str, list] = defaultdict(list)
    for r in results:
        by_lang[r["group"]].append(r)

    for lang in sorted(by_lang.keys()):
        lang_results = by_lang[lang]
        scores = [r["restoration_score"] for r in lang_results]
        # Top 5 heads by mean restoration
        head_scores: dict[tuple, list] = defaultdict(list)
        for r in lang_results:
            head_scores[(r["layer"], r["head"])].append(r["restoration_score"])
        head_means = {k: np.mean(v) for k, v in head_scores.items()}
        top5 = sorted(head_means.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"L{l}H{h}({s:.3f})" for (l, h), s in top5)
        print(f"  {lang:10s}: mean_restore={np.mean(scores):.4f}, top5: {top_str}")

    # Plots
    print("\nGenerating visualizations...")
    plot_restoration_heatmaps(results, n_layers, n_heads, out_dir)
    plot_necessity_vs_sufficiency(results, args.ablation, n_layers, n_heads, out_dir)

    print(f"\n{'=' * 60}")
    print(f"Activation patching complete! Output in {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
