"""
Attention Head Ablation Experiment — Reusable Engine

Config-driven ablation sweep for any TransformerLens-supported model.
Designed for cross-lingual and cross-task circuit comparison.

Usage:
    python ablation_experiment.py config.json
    python ablation_experiment.py config.json --device cpu --dry-run
"""

import argparse
import json
import csv
import os
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_config(cfg: dict, overrides: argparse.Namespace) -> dict:
    """Merge CLI overrides into config."""
    if overrides.device:
        cfg["device"] = overrides.device
    if overrides.model:
        cfg["model"] = overrides.model
    if overrides.output_dir:
        cfg["output_dir"] = overrides.output_dir
    if hasattr(overrides, "ablation_type") and overrides.ablation_type:
        cfg["ablation_type"] = overrides.ablation_type
    cfg.setdefault("device", "cpu")
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("ablation_type", "zero")
    cfg.setdefault("hook_point", "attn.hook_z")
    return cfg


# ── Core Engine ──────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str) -> HookedTransformer:
    print(f"  Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def resolve_targets(model: HookedTransformer, groups: dict) -> dict[str, int]:
    """Resolve target token strings to IDs across all prompt groups."""
    tokenizer = model.tokenizer
    targets = {}
    for group_name, prompts in groups.items():
        for p in prompts:
            tids = tokenizer.encode(p["target_token"], add_special_tokens=False)
            if not tids:
                print(f"  WARNING: empty tokenization for {p['id']} target '{p['target_token']}'")
                continue
            targets[p["id"]] = tids[0]
    return targets


def compute_baselines(model, all_prompts, targets):
    """Compute baseline P(target) and logit(target) for each prompt."""
    baselines = {}
    logit_baselines = {}
    for p in all_prompts:
        pid = p["id"]
        if pid not in targets:
            continue
        logits = model(p["prompt"])
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        baselines[pid] = probs[targets[pid]].item()
        logit_baselines[pid] = logits[0, -1, targets[pid]].item()
    return baselines, logit_baselines


def compute_mean_activations(model, all_prompts, targets, n_layers, n_heads,
                             hook_point="attn.hook_z"):
    """Compute mean head activations across all prompts for mean ablation.

    Mean ablation replaces head output with its mean activation (instead of zero),
    avoiding the out-of-distribution "spoofing" signal that zero ablation introduces
    (Li & Janson, NeurIPS 2024).
    """
    print("  Computing mean activations for mean ablation...")
    # Accumulate mean activations per head
    mean_acts = {}
    count = 0

    for p in all_prompts:
        pid = p["id"]
        if pid not in targets:
            continue

        _, cache = model.run_with_cache(
            p["prompt"],
            names_filter=lambda name: name.endswith(hook_point))

        for layer in range(n_layers):
            key = f"blocks.{layer}.{hook_point}"
            if key in cache:
                act = cache[key][0, -1, :, :]  # [n_heads, d_head] at last position
                if key not in mean_acts:
                    mean_acts[key] = torch.zeros_like(act)
                mean_acts[key] += act
        count += 1

    # Average
    for key in mean_acts:
        mean_acts[key] /= count

    print(f"  Mean activations computed from {count} prompts")
    return mean_acts


def ablate_head(model, text, target_id, layer, head, hook_point="attn.hook_z",
                ablation_type="zero", mean_acts=None):
    """Ablate a single head and return (prob, logit) at target.

    Supports both zero ablation and mean ablation.
    """
    if ablation_type == "mean" and mean_acts is not None:
        hook_key = f"blocks.{layer}.{hook_point}"
        mean_val = mean_acts[hook_key][head, :]

        def hook_fn(value, hook):
            value[:, :, head, :] = mean_val
            return value
    else:
        def hook_fn(value, hook):
            value[:, :, head, :] = 0.0
            return value

    hook_name = f"blocks.{layer}.{hook_point}"
    logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, hook_fn)])
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs[target_id].item(), logits[0, -1, target_id].item()


def load_checkpoint(output_dir: str) -> tuple[list[dict], set[str]]:
    """Load checkpoint if it exists. Returns (results, completed_prompt_ids)."""
    cp_path = Path(output_dir) / "checkpoint.json"
    if not cp_path.exists():
        return [], set()
    with open(cp_path) as f:
        data = json.load(f)
    completed = {r["prompt_id"] for r in data}
    print(f"  Resuming from checkpoint: {len(completed)} prompts already done")
    return data, completed


def save_checkpoint(results: list[dict], output_dir: str):
    """Append-style checkpoint: save all results so far (atomic write)."""
    cp_path = Path(output_dir) / "checkpoint.json"
    _atomic_write(cp_path, lambda f: json.dump(results, f))


def remove_checkpoint(output_dir: str):
    """Remove checkpoint file after successful completion."""
    cp_path = Path(output_dir) / "checkpoint.json"
    if cp_path.exists():
        cp_path.unlink()
        print(f"  Removed checkpoint file")


# ── Output ───────────────────────────────────────────────────────────────────

def _atomic_write(target: Path, write_fn, *, newline: str | None = None):
    """Write a file atomically: write to tmp, then os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline=newline) as f:
            write_fn(f)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, target)


def save_all(results, baselines, cfg, group_names, n_layers, n_heads, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV (atomic)
    csv_path = output_dir / "ablation_results.csv"
    fields = ["prompt_id", "group", "category", "layer", "head",
              "baseline_prob", "ablated_prob", "prob_drop",
              "baseline_logit", "ablated_logit", "logit_drop"]

    def write_csv(f):
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    _atomic_write(csv_path, write_csv, newline="")
    print(f"  {csv_path} ({len(results)} rows)")

    # Full JSON (atomic)
    json_path = output_dir / "ablation_results.json"
    payload = {
        "metadata": {
            "model": cfg["model"],
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ablation_type": cfg.get("ablation_type", "zero"),
            "hook_point": cfg.get("hook_point", "attn.hook_z"),
            "device": cfg["device"],
            "groups": group_names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    _atomic_write(json_path, lambda f: json.dump(payload, f, indent=2))
    print(f"  {json_path}")

    # Per-group summaries (atomic)
    for gname in group_names:
        g_results = [r for r in results if r["group"] == gname]
        drops = {}
        for r in g_results:
            key = f"L{r['layer']}H{r['head']}"
            drops.setdefault(key, []).append(r["prob_drop"])
        mean_drops = {k: round(np.mean(v), 6) for k, v in drops.items()}
        top = sorted(mean_drops.items(), key=lambda x: x[1], reverse=True)[:10]
        pids = {r["prompt_id"] for r in g_results}
        g_baselines = {k: round(v, 6) for k, v in baselines.items() if k in pids}

        summary = {
            "group": gname,
            "baselines": g_baselines,
            "top_10_heads": [{"head": k, "mean_prob_drop": v} for k, v in top],
        }
        path = output_dir / f"summary_{gname}.json"
        _atomic_write(path, lambda f, s=summary: json.dump(s, f, indent=2, ensure_ascii=False))
        print(f"  {path}")


def build_grid(results, n_layers, n_heads, group=None):
    filtered = [r for r in results if group is None or r["group"] == group]
    grid = np.zeros((n_layers, n_heads))
    counts = np.zeros((n_layers, n_heads))
    for r in filtered:
        grid[r["layer"], r["head"]] += r["prob_drop"]
        counts[r["layer"], r["head"]] += 1
    return grid / np.maximum(counts, 1)


def plot_single_group(results, n_layers, n_heads, group_name, output_dir):
    """Heatmap + bar plot for a single group."""
    output_dir = Path(output_dir)
    grid = build_grid(results, n_layers, n_heads, group_name)
    vmax = max(abs(grid.min()), abs(grid.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.7), max(8, n_layers * 0.4)))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(f"{group_name} — Mean P(target) Drop per Head")
    plt.colorbar(im, ax=ax, label="Mean P(target) drop")
    plt.tight_layout()
    fig.savefig(output_dir / f"heatmap_{group_name}.png", dpi=150)
    plt.close(fig)


def plot_comparison(results, n_layers, n_heads, group_names, output_dir):
    """Side-by-side heatmaps, difference map, bar chart, scatter for 2 groups."""
    output_dir = Path(output_dir)
    if len(group_names) < 2:
        return

    g1, g2 = group_names[0], group_names[1]
    grid1 = build_grid(results, n_layers, n_heads, g1)
    grid2 = build_grid(results, n_layers, n_heads, g2)
    vmax = max(abs(grid1).max(), abs(grid2).max(), 1e-6)

    # Side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(10, n_layers * 0.45)), sharey=True)
    ax1.imshow(grid1, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_title(f"{g1}", fontsize=14)
    ax1.set_xlabel("Head"); ax1.set_ylabel("Layer")
    ax1.set_xticks(range(n_heads)); ax1.set_yticks(range(n_layers))
    im = ax2.imshow(grid2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_title(f"{g2}", fontsize=14)
    ax2.set_xlabel("Head"); ax2.set_xticks(range(n_heads))
    fig.colorbar(im, ax=[ax1, ax2], label="Mean P(target) drop", shrink=0.8)
    fig.suptitle(f"Head Ablation Comparison: {g1} vs {g2}", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Difference heatmap
    diff = grid1 - grid2
    vd = max(abs(diff).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(12, max(10, n_layers * 0.45)))
    im = ax.imshow(diff, cmap="PiYG_r", vmin=-vd, vmax=vd, aspect="auto")
    ax.set_title(f"{g1} minus {g2} — Head Importance Difference", fontsize=13)
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
    plt.colorbar(im, ax=ax, label=f"Importance difference ({g1} - {g2})")
    plt.tight_layout()
    fig.savefig(output_dir / "difference_heatmap.png", dpi=150)
    plt.close(fig)

    # Top heads comparison bar chart
    labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    d1 = {labels[i]: grid1.flat[i] for i in range(len(labels))}
    d2 = {labels[i]: grid2.flat[i] for i in range(len(labels))}
    top_set = set()
    for d in [d1, d2]:
        top_set.update(k for k, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:15])
    top_heads = sorted(top_set, key=lambda h: max(d1[h], d2[h]), reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_heads) * 0.35)))
    y = np.arange(len(top_heads))
    h = 0.35
    ax.barh(y - h/2, [d1[k] for k in top_heads], h, label=g1, color="#d32f2f", alpha=0.8)
    ax.barh(y + h/2, [d2[k] for k in top_heads], h, label=g2, color="#1565c0", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(top_heads)
    ax.invert_yaxis()
    ax.set_xlabel("Mean P(target) drop")
    ax.set_title(f"Top Heads: {g1} vs {g2}")
    ax.legend(); ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "top_heads_comparison.png", dpi=150)
    plt.close(fig)

    # Correlation scatter
    v1 = [d1[l] for l in labels]
    v2 = [d2[l] for l in labels]
    corr = np.corrcoef(v1, v2)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(v2, v1, alpha=0.4, s=20, c="#555")
    for label in top_heads[:10]:
        ax.annotate(label, (d2[label], d1[label]), fontsize=8, alpha=0.8)
    lims = [min(min(v1), min(v2)) - 0.01, max(max(v1), max(v2)) + 0.01]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (shared circuit)")
    ax.set_xlabel(f"{g2} mean P(target) drop")
    ax.set_ylabel(f"{g1} mean P(target) drop")
    ax.set_title(f"Head Importance Correlation (r={corr:.3f})")
    ax.legend(); ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_scatter.png", dpi=150)
    plt.close(fig)

    return corr


def plot_all(results, n_layers, n_heads, group_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for g in group_names:
        plot_single_group(results, n_layers, n_heads, g, output_dir)
        print(f"  heatmap_{g}.png")

    corr = None
    if len(group_names) >= 2:
        corr = plot_comparison(results, n_layers, n_heads, group_names, output_dir)
        print(f"  comparison_heatmaps.png")
        print(f"  difference_heatmap.png")
        print(f"  top_heads_comparison.png")
        print(f"  correlation_scatter.png")

    return corr


# ── Dry Run ──────────────────────────────────────────────────────────────────

def dry_run(cfg, groups):
    """Print what would happen without running anything."""
    n_prompts = sum(len(v) for v in groups.values())
    # Estimate head count from known models
    known = {
        "gpt2-small": (12, 12), "gpt2": (12, 12), "gpt2-medium": (24, 16),
        "bigscience/bloom-560m": (24, 16), "bigscience/bloom-1b1": (24, 16),
    }
    nl, nh = known.get(cfg["model"], ("?", "?"))
    if nl != "?":
        total_heads = nl * nh
        total_passes = n_prompts * total_heads + n_prompts
        print(f"  Model: {cfg['model']} ({nl} layers x {nh} heads = {total_heads} heads)")
        print(f"  Prompts: {n_prompts} across {len(groups)} groups")
        print(f"  Forward passes: {total_passes} ({n_prompts} baseline + {n_prompts * total_heads} ablation)")
        print(f"  Device: {cfg['device']}")
        print(f"  Output: {cfg['output_dir']}")
    else:
        print(f"  Model: {cfg['model']} (unknown architecture — will resolve at load time)")
        print(f"  Prompts: {n_prompts} across {len(groups)} groups")
        print(f"  Device: {cfg['device']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention Head Ablation Experiment")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--device", help="Override device (cpu, cuda, mps)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--ablation-type", choices=["zero", "mean"],
                        help="Ablation method: zero (default) or mean (robustness check)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_config(cfg, args)
    groups = cfg["prompts"]
    group_names = list(groups.keys())

    print("=" * 60)
    print("Attention Head Ablation Experiment")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN]")
        dry_run(cfg, groups)
        return

    start = time.time()

    # Phase 1: Load
    print("\n[1/5] Loading model and prompts...")
    model = load_model(cfg["model"], cfg["device"])
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  {n_layers} layers x {n_heads} heads = {n_layers * n_heads} heads")

    all_prompts = [p for ps in groups.values() for p in ps]
    targets = resolve_targets(model, groups)
    print(f"  {len(all_prompts)} prompts across {len(groups)} groups: {group_names}")

    # Phase 2: Baselines
    print("\n[2/5] Computing baselines...")
    baselines, logit_baselines = compute_baselines(model, all_prompts, targets)
    for pid, prob in baselines.items():
        print(f"  {pid:25s} P={prob:.4f}  logit={logit_baselines[pid]:.4f}")

    # Compute mean activations if using mean ablation
    ablation_type = cfg.get("ablation_type", "zero")
    mean_acts = None
    if ablation_type == "mean":
        mean_acts = compute_mean_activations(
            model, all_prompts, targets, n_layers, n_heads,
            cfg.get("hook_point", "attn.hook_z")
        )

    # Phase 3: Ablation sweep (with checkpointing)
    all_results = []
    completed_pids = set()
    if args.resume:
        all_results, completed_pids = load_checkpoint(cfg["output_dir"])

    # Build flat queue of (group_name, prompt) for all remaining prompts
    sweep_queue = []
    for gname in group_names:
        for p in groups[gname]:
            if p["id"] not in completed_pids and p["id"] in targets:
                sweep_queue.append((gname, p))

    total_prompts = sum(len(g) for g in groups.values())
    total_heads = n_layers * n_heads
    hook_point = cfg.get("hook_point", "attn.hook_z")

    print(f"\n[3/5] Ablation sweep — {len(completed_pids)}/{total_prompts} already done, "
          f"{len(sweep_queue)} remaining")

    prompt_bar = tqdm(
        sweep_queue,
        desc="Prompts",
        unit="prompt",
        initial=len(completed_pids),
        total=len(completed_pids) + len(sweep_queue),
        position=0,
        dynamic_ncols=True,
    )

    for gname, p in prompt_bar:
        pid = p["id"]
        baseline = baselines[pid]
        tid = targets[pid]
        prompt_bar.set_postfix_str(f"{gname}/{pid}", refresh=False)

        prompt_results = []
        head_bar = tqdm(
            total=total_heads,
            desc=f"  Heads ({pid[:20]})",
            unit="head",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        logit_base = logit_baselines[pid]

        for layer in range(n_layers):
            for head in range(n_heads):
                ablated_prob, ablated_logit = ablate_head(
                    model, p["prompt"], tid, layer, head, hook_point,
                    ablation_type=ablation_type, mean_acts=mean_acts
                )
                prompt_results.append({
                    "prompt_id": pid,
                    "group": gname,
                    "category": p.get("category", ""),
                    "layer": layer,
                    "head": head,
                    "baseline_prob": round(baseline, 6),
                    "ablated_prob": round(ablated_prob, 6),
                    "prob_drop": round(baseline - ablated_prob, 6),
                    "baseline_logit": round(logit_base, 6),
                    "ablated_logit": round(ablated_logit, 6),
                    "logit_drop": round(logit_base - ablated_logit, 6),
                })
                head_bar.update(1)
        head_bar.close()

        all_results.extend(prompt_results)
        completed_pids.add(pid)
        save_checkpoint(all_results, cfg["output_dir"])

    # Phase 4: Save
    print("\n[4/5] Saving results...")
    save_all(all_results, baselines, cfg, group_names, n_layers, n_heads, cfg["output_dir"])

    # Remove checkpoint only after save_all succeeds
    remove_checkpoint(cfg["output_dir"])

    # Phase 5: Plot
    print("\n[5/5] Plotting...")
    corr = plot_all(all_results, n_layers, n_heads, group_names, cfg["output_dir"])

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")

    # Print summary
    print("\n" + "=" * 60)
    if corr is not None:
        print(f"Cross-group correlation: r = {corr:.3f}")
        if corr > 0.7:
            print("→ High correlation — shared circuits across groups")
        elif corr > 0.3:
            print("→ Moderate correlation — mix of shared and group-specific circuits")
        else:
            print("→ Low correlation — different circuits per group")
    print("=" * 60)

    for gname in group_names:
        path = Path(cfg["output_dir"]) / f"summary_{gname}.json"
        with open(path) as f:
            s = json.load(f)
        print(f"\nTop 5 heads for {gname}:")
        for entry in s["top_10_heads"][:5]:
            print(f"  {entry['head']:>7s}  {entry['mean_prob_drop']:+.6f}")


if __name__ == "__main__":
    main()
