"""
Cross-Lingual Attention Head Ablation: Arabic vs English in BLOOM

Compares which attention heads are critical for next-token prediction
in Arabic vs English, using the same multilingual model (BLOOM-560m).

Research question: Does BLOOM use the same circuits for both languages,
or does it develop language-specific attention heads?
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


RESULTS_DIR = Path(__file__).parent / "results" / "crosslingual"
PROMPTS_PATH = Path(__file__).parent / "prompts_crosslingual.json"


# ── Phase 1: Load ────────────────────────────────────────────────────────────

def load_prompts(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def resolve_target_ids(model, prompts_by_lang):
    target_ids = {}
    tokenizer = model.tokenizer
    for lang, prompts in prompts_by_lang.items():
        for p in prompts:
            tids = tokenizer.encode(p["target_token"], add_special_tokens=False)
            target_ids[p["id"]] = tids[0]
    return target_ids


# ── Phase 2: Baseline ────────────────────────────────────────────────────────

def get_baseline_probs(model, all_prompts, target_ids):
    baselines = {}
    for p in all_prompts:
        logits = model(p["prompt"])
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        baselines[p["id"]] = probs[target_ids[p["id"]]].item()
    return baselines


# ── Phase 3: Ablation ────────────────────────────────────────────────────────

def ablate_single_head(model, text, target_id, layer, head):
    def hook_fn(value, hook):
        value[:, :, head, :] = 0.0
        return value

    hook_name = f"blocks.{layer}.attn.hook_z"
    logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, hook_fn)])
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs[target_id].item()


def run_ablation_sweep(model, prompts, target_ids, baselines, lang_label):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = []
    total = len(prompts) * n_layers * n_heads
    done = 0

    for p in prompts:
        pid = p["id"]
        baseline = baselines[pid]
        tid = target_ids[pid]

        for layer in range(n_layers):
            for head in range(n_heads):
                ablated_prob = ablate_single_head(model, p["prompt"], tid, layer, head)
                prob_drop = baseline - ablated_prob
                results.append({
                    "prompt_id": pid,
                    "language": lang_label,
                    "category": p["category"],
                    "layer": layer,
                    "head": head,
                    "baseline_prob": round(baseline, 6),
                    "ablated_prob": round(ablated_prob, 6),
                    "prob_drop": round(prob_drop, 6),
                })
                done += 1

            pct = done / total * 100
            print(f"  [{done}/{total} {pct:.0f}%] {pid} — layer {layer} done")

    return results


# ── Phase 4: Save ────────────────────────────────────────────────────────────

def save_results(results, baselines, n_layers, n_heads):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = RESULTS_DIR / "crosslingual_results.csv"
    fieldnames = ["prompt_id", "language", "category", "layer", "head",
                  "baseline_prob", "ablated_prob", "prob_drop"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {csv_path} ({len(results)} rows)")

    # JSON
    json_path = RESULTS_DIR / "crosslingual_results.json"
    payload = {
        "metadata": {
            "model": "bigscience/bloom-560m",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ablation_type": "zero",
            "hook_point": "attn.hook_z",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {json_path}")

    # Summary per language
    for lang in ["arabic", "english"]:
        lang_results = [r for r in results if r["language"] == lang]
        head_drops = {}
        for r in lang_results:
            key = f"L{r['layer']}H{r['head']}"
            head_drops.setdefault(key, []).append(r["prob_drop"])
        mean_drops = {k: round(np.mean(v), 6) for k, v in head_drops.items()}
        top_10 = sorted(mean_drops.items(), key=lambda x: x[1], reverse=True)[:10]

        lang_baselines = {k: round(v, 6) for k, v in baselines.items()
                          if k.startswith("ar_" if lang == "arabic" else "en_")}
        summary = {
            "language": lang,
            "baselines": lang_baselines,
            "top_10_heads": [{"head": k, "mean_prob_drop": v} for k, v in top_10],
        }
        path = RESULTS_DIR / f"summary_{lang}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Saved {path}")


# ── Phase 5: Plot ────────────────────────────────────────────────────────────

def build_grid(results, n_layers, n_heads, lang):
    lang_results = [r for r in results if r["language"] == lang]
    grid = np.zeros((n_layers, n_heads))
    counts = np.zeros((n_layers, n_heads))
    for r in lang_results:
        grid[r["layer"], r["head"]] += r["prob_drop"]
        counts[r["layer"], r["head"]] += 1
    return grid / np.maximum(counts, 1)


def plot_results(results, n_layers, n_heads):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ar_grid = build_grid(results, n_layers, n_heads, "arabic")
    en_grid = build_grid(results, n_layers, n_heads, "english")

    # Shared color scale
    vmax = max(abs(ar_grid).max(), abs(en_grid).max(),
               abs(en_grid).max(), abs(ar_grid).max())

    # ── Side-by-side heatmaps ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    im1 = ax1.imshow(ar_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_title("Arabic — Mean P(target) Drop", fontsize=14)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_xticks(range(n_heads))
    ax1.set_yticks(range(n_layers))

    im2 = ax2.imshow(en_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_title("English — Mean P(target) Drop", fontsize=14)
    ax2.set_xlabel("Head")
    ax2.set_xticks(range(n_heads))

    fig.colorbar(im2, ax=[ax1, ax2], label="Mean P(target) drop", shrink=0.8)
    fig.suptitle("Cross-Lingual Head Ablation: BLOOM-560m\n"
                 "(Red = head matters, Blue = ablation helped)", fontsize=16)
    plt.tight_layout()
    path = RESULTS_DIR / "crosslingual_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Difference heatmap (Arabic - English) ──
    diff_grid = ar_grid - en_grid
    vmax_diff = abs(diff_grid).max()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(diff_grid, cmap="PiYG_r", vmin=-vmax_diff, vmax=vmax_diff, aspect="auto")
    ax.set_title("Arabic minus English Head Importance\n"
                 "(Pink = more important for Arabic, Green = more important for English)",
                 fontsize=13)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    plt.colorbar(im, ax=ax, label="Importance difference (Arabic - English)")
    plt.tight_layout()
    path = RESULTS_DIR / "language_difference_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Top heads comparison bar plot ──
    ar_drops = {}
    en_drops = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            label = f"L{layer}H{head}"
            ar_drops[label] = ar_grid[layer, head]
            en_drops[label] = en_grid[layer, head]

    # Get top 15 heads from either language
    all_heads = set()
    for drops in [ar_drops, en_drops]:
        top = sorted(drops.items(), key=lambda x: x[1], reverse=True)[:15]
        all_heads.update(k for k, v in top)
    all_heads = sorted(all_heads, key=lambda h: max(ar_drops[h], en_drops[h]), reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(all_heads))
    height = 0.35
    ax.barh(y - height/2, [ar_drops[h] for h in all_heads], height, label="Arabic", color="#d32f2f", alpha=0.8)
    ax.barh(y + height/2, [en_drops[h] for h in all_heads], height, label="English", color="#1565c0", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_heads)
    ax.invert_yaxis()
    ax.set_xlabel("Mean P(target) drop")
    ax.set_title("Top Attention Heads: Arabic vs English")
    ax.legend()
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = RESULTS_DIR / "top_heads_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Correlation scatter plot ──
    all_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    ar_vals = [ar_drops[h] for h in all_labels]
    en_vals = [en_drops[h] for h in all_labels]

    corr = np.corrcoef(ar_vals, en_vals)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(en_vals, ar_vals, alpha=0.4, s=20, c="#555")

    # Label top heads
    for label in all_heads[:10]:
        ax.annotate(label, (en_drops[label], ar_drops[label]),
                    fontsize=8, alpha=0.8)

    lims = [min(min(ar_vals), min(en_vals)) - 0.02,
            max(max(ar_vals), max(en_vals)) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (shared circuit)")
    ax.set_xlabel("English mean P(target) drop")
    ax.set_ylabel("Arabic mean P(target) drop")
    ax.set_title(f"Head Importance: Arabic vs English (r={corr:.3f})")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path = RESULTS_DIR / "correlation_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return corr


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Cross-Lingual Head Ablation: Arabic vs English")
    print("Model: BLOOM-560m (bigscience/bloom-560m)")
    print("=" * 60)
    start = time.time()

    # Phase 1
    print("\n[Phase 1] Loading model and prompts...")
    model = HookedTransformer.from_pretrained("bigscience/bloom-560m", device="cpu")
    model.eval()
    torch.set_grad_enabled(False)

    prompts_by_lang = load_prompts(PROMPTS_PATH)
    target_ids = resolve_target_ids(model, prompts_by_lang)
    all_prompts = prompts_by_lang["arabic"] + prompts_by_lang["english"]

    print(f"  {len(prompts_by_lang['arabic'])} Arabic + {len(prompts_by_lang['english'])} English prompts")
    print(f"  Model: {model.cfg.n_layers} layers x {model.cfg.n_heads} heads = "
          f"{model.cfg.n_layers * model.cfg.n_heads} heads")

    for p in all_prompts:
        print(f"    {p['id']}: target → token_id {target_ids[p['id']]}")

    # Phase 2
    print("\n[Phase 2] Baseline inference...")
    baselines = get_baseline_probs(model, all_prompts, target_ids)
    print("  Baselines:")
    for pid, prob in baselines.items():
        print(f"    {pid:20s}: {prob:.4f}")

    # Phase 3
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print(f"\n[Phase 3a] Arabic ablation sweep...")
    ar_results = run_ablation_sweep(
        model, prompts_by_lang["arabic"], target_ids, baselines, "arabic")

    print(f"\n[Phase 3b] English ablation sweep...")
    en_results = run_ablation_sweep(
        model, prompts_by_lang["english"], target_ids, baselines, "english")

    all_results = ar_results + en_results

    # Phase 4
    print("\n[Phase 4] Saving results...")
    save_results(all_results, baselines, n_layers, n_heads)

    # Phase 5
    print("\n[Phase 5] Plotting results...")
    corr = plot_results(all_results, n_layers, n_heads)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")

    # Summary
    print("\n" + "=" * 60)
    print(f"Pearson correlation of head importance (Arabic vs English): r = {corr:.3f}")
    print("=" * 60)

    if corr > 0.7:
        print("High correlation — BLOOM largely uses SHARED circuits")
    elif corr > 0.3:
        print("Moderate correlation — mix of shared and language-specific circuits")
    else:
        print("Low correlation — BLOOM uses DIFFERENT circuits per language")

    for lang in ["arabic", "english"]:
        path = RESULTS_DIR / f"summary_{lang}.json"
        with open(path) as f:
            summary = json.load(f)
        print(f"\nTop 10 heads for {lang.upper()}:")
        for entry in summary["top_10_heads"]:
            print(f"  {entry['head']:>7s}  →  {entry['mean_prob_drop']:+.6f}")


if __name__ == "__main__":
    main()
