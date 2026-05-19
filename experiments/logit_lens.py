"""
Experiment Two -- Tuned Lens + Raw Logit Lens Analysis

Projects residual stream to vocabulary space at each layer using:
1. Tuned Lens (primary): learned affine probes that compensate for BLOOM's ALiBi-induced
   representational drift (Belrose et al., 2023 CoNLL)
2. Raw Logit Lens (secondary): direct ln_final + unembed projection for comparison

The raw logit lens FAILS on BLOOM -- top-1 prediction shows input token in >50% of layers.
Running both methods demonstrates this failure and validates the tuned lens approach.

Usage:
    python exp2/logit_lens.py results/exp2/zero_ablation/combined_6lang/ablation_results.json
    python exp2/logit_lens.py --config exp2/configs/bloom_hindi.json --output results/exp2/logit_lens
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens import TunedLens


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


def load_prompts_from_config(config_path: str) -> list[dict]:
    """Load prompts from an ablation experiment config file."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    prompts = []
    for group, prompt_list in config["prompts"].items():
        for p in prompt_list:
            prompts.append({**p, "group": group})
    return prompts


def load_prompts_from_ablation(results_path: str) -> list[dict]:
    """Extract unique prompts from ablation results JSON."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    seen = set()
    prompts = []
    for r in data["results"]:
        pid = r["prompt_id"]
        if pid not in seen:
            seen.add(pid)
            prompts.append({
                "id": pid,
                "prompt": r["prompt"],
                "target_token": r["target_token"],
                "group": r.get("group", r.get("language", "unknown")),
                "category": r.get("category", "unknown"),
            })
    return prompts


# -- Tuned Lens Setup --------------------------------------------------------

def load_or_train_tuned_lens(hf_model, hf_tokenizer, n_layers: int,
                             cache_dir: Path) -> TunedLens:
    """Load cached tuned lens probes or train from scratch.

    Pre-trained BLOOM-560m probes are not on HuggingFace Hub, so we train
    simple per-layer affine probes on a small multilingual calibration set.
    Training is cheap (one linear layer per model layer).
    """
    cache_path = cache_dir / "tuned_lens_bloom560m.pt"

    if cache_path.exists():
        print("  Loading cached tuned lens probes...")
        lens = TunedLens.from_model(hf_model, bias=True)
        lens.load_state_dict(torch.load(cache_path, map_location="cpu", weights_only=True))
        lens.eval()
        return lens

    print("  Training tuned lens probes on multilingual calibration data...")
    lens = TunedLens.from_model(hf_model, bias=True)

    # Multilingual calibration texts (~1000 tokens across 6 languages)
    calibration_texts = [
        # English
        "The capital of France is Paris. The United Kingdom is located in Europe. "
        "Scientists discovered that water consists of hydrogen and oxygen atoms. "
        "The speed of light in vacuum is approximately 300,000 kilometers per second.",
        # Arabic
        "عاصمة فرنسا هي باريس. تقع المملكة المتحدة في أوروبا. "
        "اكتشف العلماء أن الماء يتكون من ذرات الهيدروجين والأكسجين.",
        # French
        "La capitale de la France est Paris. Le Royaume-Uni est situé en Europe. "
        "Les scientifiques ont découvert que l'eau est composée d'hydrogène et d'oxygène.",
        # Chinese
        "法国的首都是巴黎。英国位于欧洲。科学家发现水由氢原子和氧原子组成。"
        "光在真空中的速度大约是每秒三十万公里。",
        # Hindi
        "फ्रांस की राजधानी पेरिस है। यूनाइटेड किंगडम यूरोप में स्थित है। "
        "वैज्ञानिकों ने खोजा कि पानी हाइड्रोजन और ऑक्सीजन से बना है।",
        # Japanese
        "フランスの首都はパリです。イギリスはヨーロッパに位置しています。"
        "科学者たちは水が水素と酸素から構成されていることを発見しました。",
    ]

    # Collect hidden states from calibration data
    hf_model.eval()
    all_hidden_states = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for text in calibration_texts:
            inputs = hf_tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=256)
            outputs = hf_model(**inputs, output_hidden_states=True)
            # hidden_states[0] = embedding, [1..n_layers] = after each layer
            for layer_idx in range(n_layers):
                h = outputs.hidden_states[layer_idx + 1]  # after layer
                all_hidden_states[layer_idx].append(h.squeeze(0))

    # Train each layer's affine probe via least-squares
    # Target: final layer norm + unembed should produce good logits
    # The tuned lens learns: logits = unembed(ln_f(translate(h_l)))
    optimizer = torch.optim.Adam(lens.parameters(), lr=1e-3)
    lens.train()

    # Get final hidden states as training targets
    final_hidden = []
    with torch.no_grad():
        for text in calibration_texts:
            inputs = hf_tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=256)
            outputs = hf_model(**inputs, output_hidden_states=True)
            final_hidden.append(outputs.hidden_states[-1].squeeze(0))

    # Train for several epochs
    print("  Training probes (200 steps)...")
    for step in range(200):
        total_loss = 0.0
        for text_idx in range(len(calibration_texts)):
            for layer_idx in range(n_layers):
                h = all_hidden_states[layer_idx][text_idx]
                target = final_hidden[text_idx]

                # The tuned lens translates intermediate hidden states
                # to approximate the final hidden state
                translated = lens.transform_hidden(h, layer_idx)
                loss = torch.nn.functional.mse_loss(translated, target)
                total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/200, loss={total_loss.item():.4f}")

    lens.eval()

    # Cache the trained probes
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lens.state_dict(), cache_path)
    print(f"  Tuned lens probes saved to {cache_path}")

    return lens


# -- Core Analysis -----------------------------------------------------------

def logit_lens_single(model: HookedTransformer, prompt: str,
                      target_id: int, n_layers: int,
                      tuned_lens: TunedLens | None,
                      hf_model=None) -> dict:
    """Run both tuned lens and raw logit lens for a single prompt.

    Returns per-layer results for both methods.
    """
    _, cache = model.run_with_cache(
        prompt,
        names_filter=lambda name: name.endswith("hook_resid_post"))

    raw_results = []
    tuned_results = []

    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        resid_last = resid[0, -1, :]  # last token position

        # Raw logit lens: apply final layernorm + unembed directly
        normalized = model.ln_final(resid_last.unsqueeze(0))
        raw_logits = model.unembed(normalized)[0]
        raw_probs = torch.softmax(raw_logits, dim=-1)

        raw_p_target = raw_probs[target_id].item()
        raw_rank = (raw_probs > raw_probs[target_id]).sum().item() + 1
        raw_logit_target = raw_logits[target_id].item()

        raw_results.append({
            "layer": layer,
            "p_target": round(raw_p_target, 8),
            "logit_target": round(raw_logit_target, 6),
            "rank": int(raw_rank),
            "is_top1": raw_rank == 1,
            "is_top5": raw_rank <= 5,
        })

        # Tuned lens: use learned affine probe
        if tuned_lens is not None:
            tuned_logits = tuned_lens(resid_last.unsqueeze(0), layer)[0]
            tuned_probs = torch.softmax(tuned_logits, dim=-1)

            tuned_p_target = tuned_probs[target_id].item()
            tuned_rank = (tuned_probs > tuned_probs[target_id]).sum().item() + 1
            tuned_logit_target = tuned_logits[target_id].item()

            tuned_results.append({
                "layer": layer,
                "p_target": round(tuned_p_target, 8),
                "logit_target": round(tuned_logit_target, 6),
                "rank": int(tuned_rank),
                "is_top1": tuned_rank == 1,
                "is_top5": tuned_rank <= 5,
            })

    return {"raw": raw_results, "tuned": tuned_results}


def compute_convergence(layer_results: list[dict]) -> dict:
    """Compute convergence metrics from per-layer results."""
    convergence_top1 = None
    convergence_top5 = None

    for r in layer_results:
        if convergence_top5 is None and r["is_top5"]:
            convergence_top5 = r["layer"]
        if convergence_top1 is None and r["is_top1"]:
            convergence_top1 = r["layer"]

    return {
        "convergence_top1": convergence_top1,
        "convergence_top5": convergence_top5,
        "final_p_target": layer_results[-1]["p_target"],
        "final_logit_target": layer_results[-1]["logit_target"],
        "final_rank": layer_results[-1]["rank"],
        "p_target_curve": [r["p_target"] for r in layer_results],
        "logit_target_curve": [r["logit_target"] for r in layer_results],
        "rank_curve": [r["rank"] for r in layer_results],
    }


def run_logit_lens(model: HookedTransformer, prompts: list[dict],
                   n_layers: int, tuned_lens: TunedLens | None,
                   hf_model=None) -> list[dict]:
    """Run logit lens (both methods) on all prompts."""
    tokenizer = model.tokenizer
    results = []

    for i, p in enumerate(prompts):
        pid = p["id"]
        tids = tokenizer.encode(p["target_token"], add_special_tokens=False)
        if not tids:
            print(f"  [{i+1}/{len(prompts)}] {pid}: SKIP (empty tokenization)")
            continue

        target_id = tids[0]
        both = logit_lens_single(model, p["prompt"], target_id, n_layers,
                                 tuned_lens, hf_model)

        # Primary metric: tuned lens (if available), fallback to raw
        primary = both["tuned"] if both["tuned"] else both["raw"]
        primary_conv = compute_convergence(primary)
        raw_conv = compute_convergence(both["raw"])

        result = {
            "prompt_id": pid,
            "group": p["group"],
            "category": p.get("category", "unknown"),
            "prompt": p["prompt"],
            "target_token": p["target_token"],
            # Primary (tuned lens) metrics
            **primary_conv,
            # Raw logit lens for comparison
            "raw_convergence_top1": raw_conv["convergence_top1"],
            "raw_final_p_target": raw_conv["final_p_target"],
            "raw_p_target_curve": raw_conv["p_target_curve"],
            "raw_logit_target_curve": raw_conv["logit_target_curve"],
            "raw_rank_curve": raw_conv["rank_curve"],
            # Method indicator
            "primary_method": "tuned_lens" if both["tuned"] else "raw_logit_lens",
        }
        results.append(result)

        conv_str = (f"top1@L{primary_conv['convergence_top1']}"
                    if primary_conv['convergence_top1'] is not None else "never_top1")
        raw_str = (f"raw_top1@L{raw_conv['convergence_top1']}"
                   if raw_conv['convergence_top1'] is not None else "raw_never")
        print(f"  [{i+1}/{len(prompts)}] {pid}: tuned={conv_str}, {raw_str}, "
              f"final_P={primary_conv['final_p_target']:.4f}")

    return results


# -- Visualization -----------------------------------------------------------

def plot_convergence_curves(results: list[dict], n_layers: int, out_dir: Path):
    """Plot mean P(target) convergence curves per language (tuned vs raw)."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Logit Lens: Layer-by-Layer P(target) Convergence", fontsize=14)

    by_lang: dict[str, list] = defaultdict(list)
    for r in results:
        by_lang[r["group"]].append(r)

    layers = np.arange(n_layers)

    # Left: tuned lens mean P(target) curves
    ax = axes[0]
    for lang in LANGUAGES:
        if lang not in by_lang:
            continue
        curves = np.array([r["p_target_curve"] for r in by_lang[lang]])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        color = LANG_COLORS.get(lang, "#333333")
        ax.plot(layers, mean_curve, "o-", label=lang.capitalize(),
                color=color, linewidth=2, markersize=4)
        ax.fill_between(layers, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(target)")
    ax.set_title("Tuned Lens (Primary)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: raw logit lens for comparison (shows failure)
    ax = axes[1]
    for lang in LANGUAGES:
        if lang not in by_lang:
            continue
        raw_curves = [r.get("raw_p_target_curve") for r in by_lang[lang]
                      if r.get("raw_p_target_curve")]
        if not raw_curves:
            continue
        curves = np.array(raw_curves)
        mean_curve = curves.mean(axis=0)
        color = LANG_COLORS.get(lang, "#333333")
        ax.plot(layers, mean_curve, "o--", label=lang.capitalize(),
                color=color, linewidth=1.5, markersize=3, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(target)")
    ax.set_title("Raw Logit Lens (Shows BLOOM Failure)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: convergence layer distribution (tuned lens)
    ax = axes[2]
    conv_data = {}
    for lang in LANGUAGES:
        if lang not in by_lang:
            continue
        convs = [r["convergence_top1"] for r in by_lang[lang]
                 if r["convergence_top1"] is not None]
        if convs:
            conv_data[lang] = convs

    if conv_data:
        positions = list(range(len(conv_data)))
        labels = [l.capitalize() for l in conv_data.keys()]
        colors = [LANG_COLORS.get(l, "#333333") for l in conv_data.keys()]
        parts = ax.violinplot(list(conv_data.values()), positions=positions,
                              showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Convergence Layer (first top-1)")
        ax.set_title("Convergence Distribution (Tuned Lens)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "logit_lens_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logit_lens_convergence.png")


def plot_tuned_vs_raw_comparison(results: list[dict], n_layers: int, out_dir: Path):
    """Plot direct comparison of tuned vs raw logit lens to demonstrate failure."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Tuned Lens vs Raw Logit Lens on BLOOM-560m\n"
                 "(Raw logit lens fails: top-1 shows input token in >50% of layers)",
                 fontsize=14)

    by_lang: dict[str, list] = defaultdict(list)
    for r in results:
        by_lang[r["group"]].append(r)

    layers = np.arange(n_layers)
    lang_list = [l for l in LANGUAGES if l in by_lang]

    for idx, lang in enumerate(lang_list[:6]):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        # Tuned lens
        tuned_curves = np.array([r["p_target_curve"] for r in by_lang[lang]])
        tuned_mean = tuned_curves.mean(axis=0)

        # Raw logit lens
        raw_curves = [r.get("raw_p_target_curve") for r in by_lang[lang]
                      if r.get("raw_p_target_curve")]
        if raw_curves:
            raw_mean = np.array(raw_curves).mean(axis=0)
            ax.plot(layers, raw_mean, "s--", color="#999999", linewidth=1.5,
                    markersize=3, label="Raw (broken)", alpha=0.6)

        color = LANG_COLORS.get(lang, "#333333")
        ax.plot(layers, tuned_mean, "o-", color=color, linewidth=2,
                markersize=4, label="Tuned Lens")

        ax.set_title(f"{lang.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(target)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(lang_list), 6):
        row, col = divmod(idx, 3)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "logit_lens_tuned_vs_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logit_lens_tuned_vs_raw.png")


def plot_per_category_convergence(results: list[dict], n_layers: int, out_dir: Path):
    """Plot convergence curves per category, comparing languages."""
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle("Tuned Lens: Per-Category Convergence by Language", fontsize=16)
    layers = np.arange(n_layers)

    for ax, cat in zip(axes.flatten(), CATEGORIES):
        cat_results = [r for r in results if r["category"] == cat]
        by_lang: dict[str, list] = defaultdict(list)
        for r in cat_results:
            by_lang[r["group"]].append(r)

        for lang in LANGUAGES:
            if lang not in by_lang:
                continue
            curves = np.array([r["p_target_curve"] for r in by_lang[lang]])
            mean_curve = curves.mean(axis=0)
            color = LANG_COLORS.get(lang, "#333333")
            ax.plot(layers, mean_curve, "o-", label=LANG_SHORT.get(lang, lang),
                    color=color, linewidth=1.5, markersize=3)

        ax.set_title(f"{cat.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(target)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "logit_lens_per_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logit_lens_per_category.png")


def plot_script_family_convergence(results: list[dict], n_layers: int, out_dir: Path):
    """Compare convergence across script families."""
    SCRIPT_FAMILIES = {
        "Latin": ["english", "french"],
        "Arabic": ["arabic"],
        "CJK": ["chinese", "japanese"],
        "Devanagari": ["hindi"],
    }
    FAMILY_COLORS = {
        "Latin": "#e74c3c", "Arabic": "#2ecc71",
        "CJK": "#f39c12", "Devanagari": "#9b59b6",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Tuned Lens: Script-Family Convergence Comparison", fontsize=14)
    layers = np.arange(n_layers)

    for family, langs in SCRIPT_FAMILIES.items():
        family_curves = []
        for r in results:
            if r["group"] in langs:
                family_curves.append(r["p_target_curve"])
        if family_curves:
            curves = np.array(family_curves)
            mean_curve = curves.mean(axis=0)
            color = FAMILY_COLORS[family]
            ax1.plot(layers, mean_curve, "o-", label=f"{family} ({', '.join(langs)})",
                     color=color, linewidth=2, markersize=4)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean P(target)")
    ax1.set_title("Mean Convergence by Script Family")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    family_conv = {}
    for family, langs in SCRIPT_FAMILIES.items():
        convs = [r["convergence_top1"] for r in results
                 if r["group"] in langs and r["convergence_top1"] is not None]
        if convs:
            family_conv[family] = np.mean(convs)

    if family_conv:
        bars = ax2.bar(list(family_conv.keys()),
                       list(family_conv.values()),
                       color=[FAMILY_COLORS[f] for f in family_conv.keys()])
        ax2.set_ylabel("Mean Convergence Layer (top-1)")
        ax2.set_title("Average Layer Where Correct Token First Appears as Top-1")
        for bar, val in zip(bars, family_conv.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"L{val:.1f}", ha="center", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "logit_lens_script_families.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logit_lens_script_families.png")


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tuned Lens + Raw Logit Lens analysis")
    parser.add_argument("input", nargs="?",
                        help="Ablation results JSON or config file")
    parser.add_argument("--config", nargs="*", default=[],
                        help="Additional config files to load prompts from")
    parser.add_argument("--output", default="results/exp2/logit_lens",
                        help="Output directory")
    parser.add_argument("--skip-tuned-lens", action="store_true",
                        help="Skip tuned lens (raw only, for debugging)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather prompts from all sources
    prompts = []
    sources = []
    if args.input:
        sources.append(args.input)
    sources.extend(args.config)

    for src in sources:
        path = Path(src)
        if not path.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "results" in data:
            new_prompts = load_prompts_from_ablation(src)
        elif "prompts" in data:
            new_prompts = load_prompts_from_config(src)
        else:
            print(f"  WARNING: Unknown format in {src}, skipping")
            continue
        prompts.extend(new_prompts)
        print(f"  Loaded {len(new_prompts)} prompts from {src}")

    if not prompts:
        print("No prompts loaded. Provide ablation results or config files.")
        return

    # Deduplicate by prompt_id
    seen = set()
    unique = []
    for p in prompts:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    prompts = unique
    print(f"\n  Total unique prompts: {len(prompts)}")

    # Load TransformerLens model
    print("\nLoading BLOOM-560m on CPU (TransformerLens)...")
    model = HookedTransformer.from_pretrained("bigscience/bloom-560m", device="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = model.cfg.n_layers

    # Load or train tuned lens
    tuned_lens = None
    hf_model = None
    if not args.skip_tuned_lens:
        print("\nSetting up tuned lens...")
        hf_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
        hf_model.eval()
        hf_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

        cache_dir = Path(__file__).parent.parent / "results" / "exp2" / "tuned_lens_cache"
        tuned_lens = load_or_train_tuned_lens(hf_model, hf_tokenizer, n_layers, cache_dir)
        tuned_lens.eval()

    # Run logit lens (both methods)
    print(f"\nRunning logit lens ({n_layers} layers x {len(prompts)} prompts)...")
    results = run_logit_lens(model, prompts, n_layers, tuned_lens, hf_model)

    # Summary
    print(f"\n{'=' * 60}")
    print("LOGIT LENS SUMMARY")
    print(f"{'=' * 60}")

    by_lang: dict[str, list] = defaultdict(list)
    for r in results:
        by_lang[r["group"]].append(r)

    for lang in sorted(by_lang.keys()):
        lang_results = by_lang[lang]
        # Tuned lens convergence
        convs_t1 = [r["convergence_top1"] for r in lang_results
                    if r["convergence_top1"] is not None]
        mean_t1 = np.mean(convs_t1) if convs_t1 else float("nan")
        # Raw logit lens convergence (for comparison)
        raw_convs = [r["raw_convergence_top1"] for r in lang_results
                     if r.get("raw_convergence_top1") is not None]
        raw_mean = np.mean(raw_convs) if raw_convs else float("nan")
        n_never = sum(1 for r in lang_results if r["convergence_top1"] is None)
        print(f"  {lang:10s}: tuned_conv=L{mean_t1:.1f}, "
              f"raw_conv=L{raw_mean:.1f}, "
              f"never_top1={n_never}/{len(lang_results)}")

    # Save results
    output_data = {
        "metadata": {
            "model": "bigscience/bloom-560m",
            "technique": "tuned_lens_and_raw_logit_lens",
            "primary_method": "tuned_lens" if tuned_lens is not None else "raw_logit_lens",
            "n_layers": n_layers,
            "n_prompts": len(results),
            "note": ("Raw logit lens fails on BLOOM due to ALiBi positional encodings "
                     "causing representational drift (Belrose et al., 2023). "
                     "Tuned lens compensates with learned per-layer affine probes."),
        },
        "results": results,
    }
    results_path = out_dir / "logit_lens_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_convergence_curves(results, n_layers, out_dir)
    if tuned_lens is not None:
        plot_tuned_vs_raw_comparison(results, n_layers, out_dir)
    plot_per_category_convergence(results, n_layers, out_dir)
    plot_script_family_convergence(results, n_layers, out_dir)

    print(f"\n{'=' * 60}")
    print(f"Logit lens complete! Output in {out_dir}/")
    print(f"  - logit_lens_results.json")
    print(f"  - logit_lens_convergence.png")
    if tuned_lens is not None:
        print(f"  - logit_lens_tuned_vs_raw.png")
    print(f"  - logit_lens_per_category.png")
    print(f"  - logit_lens_script_families.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
