#!/usr/bin/env python3
"""Extended analysis: temperature sensitivity, score logit distributions,
cross-attention, multi-prompt encoding robustness, gradient attribution details.
All results saved to extended_stats.json for manuscript cross-reference.

This script reads the raw per-image result shards from
`results/runpod_final/`. Those shards are NOT redistributed in this GitHub
repository; they are deposited in the companion Zenodo archive. Pre-computed
outputs are already available as data/extended_stats.json.
"""

import json, sys, warnings
import numpy as np
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
STAT_FILE = ROOT / "data" / "extended_stats.json"

if not (RESULTS / "runpod_final").exists():
    sys.exit(
        "Per-image result shards required. They are NOT in this GitHub "
        "repository. Download the Zenodo archive and place its "
        "derived_results/runpod_final/ directory at "
        f"{(RESULTS / 'runpod_final').relative_to(ROOT)}. "
        "Pre-computed outputs are in data/ and figures/."
    )

FIGDIR.mkdir(parents=True, exist_ok=True)

CC = {"korean": "#E74C3C", "chinese": "#E67E22", "japanese": "#2ECC71", "western": "#3498DB"}
CO = ["korean", "chinese", "japanese", "western"]
CL = {"korean": "Korean", "chinese": "Chinese", "japanese": "Japanese", "western": "Western"}
EA = ["korean", "chinese", "japanese"]

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def sf(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except:
        return None

def d_eff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    p = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / p) if p > 1e-10 else 0.0

def load_all():
    def _load(prefix):
        recs = []
        for i in range(4):
            p = RESULTS / "runpod_final" / f"{prefix}_shard{i}" / "results.json"
            try:
                with open(p) as f: recs.extend(json.load(f))
            except: pass
        return recs
    return _load("qwen3vl"), _load("llama32v")

def by_r(recs):
    return {"EA": [r for r in recs if r.get("culture") in EA],
            "W":  [r for r in recs if r.get("culture") == "western"]}

def by_c(recs):
    return {c: [r for r in recs if r.get("culture") == c] for c in CO}

all_stats = {}

# ═══════════════════════════════════════════════════════════════
def analyze_temperature(qwen, llama):
    """Temperature sensitivity: is score gap stable across temperatures?"""
    print("=" * 60)
    print("  TEMPERATURE SENSITIVITY")
    print("=" * 60)

    temps = ["temp_0.0", "temp_0.5", "temp_1.0"]
    temp_labels = ["T=0.0", "T=0.5", "T=1.0"]
    temp_stats = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (mname, recs) in zip(axes, [("Qwen", qwen), ("Llama", llama)]):
        model_stats = {}
        ea_means, w_means, ds = [], [], []
        for t, tl in zip(temps, temp_labels):
            ea_scores, w_scores = [], []
            for r in recs:
                ts = r.get("temperature_sensitivity")
                if not ts or t not in ts:
                    continue
                sc = sf(ts[t].get("score")) if isinstance(ts[t], dict) else sf(ts[t])
                if sc is None:
                    continue
                if r.get("culture") in EA:
                    ea_scores.append(sc)
                elif r.get("culture") == "western":
                    w_scores.append(sc)

            if ea_scores and w_scores:
                d = d_eff(ea_scores, w_scores)
                _, p = stats.ttest_ind(ea_scores, w_scores)
                ea_m, w_m = float(np.mean(ea_scores)), float(np.mean(w_scores))
                model_stats[tl] = {
                    "ea_mean": ea_m, "w_mean": w_m,
                    "ea_n": len(ea_scores), "w_n": len(w_scores),
                    "d": d, "p": float(p)
                }
                ea_means.append(ea_m)
                w_means.append(w_m)
                ds.append(d)
                print(f"  [{mname}] {tl}: EA={ea_m:.3f} (n={len(ea_scores)}), "
                      f"W={w_m:.3f} (n={len(w_scores)}), d={d:+.3f}, p={p:.2e}")

        temp_stats[mname.lower()] = model_stats

        x = np.arange(len(temp_labels))
        w = 0.35
        bars_ea = ax.bar(x - w/2, ea_means, w, color="#E74C3C", alpha=0.7, label="East Asian")
        bars_w = ax.bar(x + w/2, w_means, w, color="#3498DB", alpha=0.7, label="Western")
        for i, d in enumerate(ds):
            ax.text(i, max(ea_means[i], w_means[i]) + 0.05, f"d={d:+.2f}",
                    ha='center', fontsize=8, fontstyle='italic')
        ax.set_xticks(x)
        ax.set_xticklabels(temp_labels)
        ax.set_ylabel("Mean Score")
        ax.set_title(mname, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(5.5, 9.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle("Score Stability Across Temperatures", fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig_temperature_sensitivity.pdf")
    fig.savefig(FIGDIR / "fig_temperature_sensitivity.png")
    plt.close(fig)
    all_stats["temperature"] = temp_stats
    print("  Saved fig_temperature_sensitivity")


# ═══════════════════════════════════════════════════════════════
def analyze_logit_distribution(qwen, llama):
    """Score logit distribution: shape differences EA vs W."""
    print("\n" + "=" * 60)
    print("  SCORE LOGIT DISTRIBUTION")
    print("=" * 60)

    logit_stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (mname, recs) in zip(axes, [("Qwen", qwen), ("Llama", llama)]):
        regs = by_r(recs)
        model_stats = {}

        for region, label, color in [("EA", "East Asian", "#E74C3C"), ("W", "Western", "#3498DB")]:
            score_bins = {str(i): [] for i in range(1, 11)}
            for r in regs[region]:
                ld = r.get("aesthetic_score_logit_distribution")
                if not ld or not isinstance(ld, dict):
                    continue
                for k in range(1, 11):
                    v = sf(ld.get(str(k)))
                    if v is not None:
                        score_bins[str(k)].append(v)

            if all(len(score_bins[str(k)]) > 0 for k in range(1, 11)):
                means = [float(np.mean(score_bins[str(k)])) for k in range(1, 11)]
                sems = [float(np.std(score_bins[str(k)]) / np.sqrt(len(score_bins[str(k)])))
                        for k in range(1, 11)]
                n = len(score_bins["1"])
                x = np.arange(1, 11)
                ax.plot(x, means, 'o-', color=color, label=f"{label} (n={n})", lw=1.5, markersize=4)
                ax.fill_between(x, np.array(means) - 1.96*np.array(sems),
                               np.array(means) + 1.96*np.array(sems), color=color, alpha=0.1)

                model_stats[region] = {
                    "n": n,
                    "means": means,
                    "peak_score": int(np.argmax(means) + 1),
                    "mean_of_means": float(np.mean(means)),
                }

                peak_ea = model_stats.get("EA", {}).get("peak_score")
                peak_w = model_stats.get("W", {}).get("peak_score")

        if "EA" in model_stats and "W" in model_stats:
            ea_peaks = model_stats["EA"]["means"]
            w_peaks = model_stats["W"]["means"]
            kl_div = float(np.sum([
                p * np.log(p / q) if p > 0 and q > 0 else 0
                for p, q in zip(
                    np.array(ea_peaks) / np.sum(ea_peaks),
                    np.array(w_peaks) / np.sum(w_peaks)
                )
            ]))
            model_stats["kl_ea_w"] = kl_div
            print(f"  [{mname}] EA peak={model_stats['EA']['peak_score']}, "
                  f"W peak={model_stats['W']['peak_score']}, KL(EA||W)={kl_div:.4f}")

        ax.set_xlabel("Score (1-10)")
        ax.set_ylabel("Mean Logit")
        ax.set_title(mname, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        logit_stats[mname.lower()] = model_stats

    fig.suptitle("Score Logit Distributions: EA vs Western", fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig_logit_distribution.pdf")
    fig.savefig(FIGDIR / "fig_logit_distribution.png")
    plt.close(fig)
    all_stats["logit_distribution"] = logit_stats
    print("  Saved fig_logit_distribution")


# ═══════════════════════════════════════════════════════════════
def analyze_cross_attention(llama):
    """Cross-attention analysis (Llama only)."""
    print("\n" + "=" * 60)
    print("  CROSS-ATTENTION (Llama)")
    print("=" * 60)

    regs = by_r(llama)
    ca_stats = {}

    ea_ent = [r["aesthetic_cross_attn_mean_entropy"] for r in regs["EA"]
              if sf(r.get("aesthetic_cross_attn_mean_entropy")) is not None]
    w_ent = [r["aesthetic_cross_attn_mean_entropy"] for r in regs["W"]
             if sf(r.get("aesthetic_cross_attn_mean_entropy")) is not None]

    if ea_ent and w_ent:
        d = d_eff(ea_ent, w_ent)
        _, p = stats.ttest_ind(ea_ent, w_ent)
        ca_stats["mean_entropy"] = {
            "ea_mean": float(np.mean(ea_ent)), "w_mean": float(np.mean(w_ent)),
            "ea_n": len(ea_ent), "w_n": len(w_ent),
            "d": d, "p": float(p)
        }
        print(f"  Mean cross-attn entropy: EA={np.mean(ea_ent):.4f}, W={np.mean(w_ent):.4f}, "
              f"d={d:+.3f}, p={p:.2e}")

    # Per-layer cross-attention entropy
    ea_layers, w_layers = [], []
    for r in regs["EA"]:
        ph = r.get("aesthetic_cross_attn_per_head_entropy")
        if ph and isinstance(ph, list) and len(ph) > 0:
            layer_means = [float(np.mean(layer)) for layer in ph if isinstance(layer, list)]
            if layer_means:
                ea_layers.append(layer_means)
    for r in regs["W"]:
        ph = r.get("aesthetic_cross_attn_per_head_entropy")
        if ph and isinstance(ph, list) and len(ph) > 0:
            layer_means = [float(np.mean(layer)) for layer in ph if isinstance(layer, list)]
            if layer_means:
                w_layers.append(layer_means)

    if ea_layers and w_layers:
        min_len = min(min(len(l) for l in ea_layers), min(len(l) for l in w_layers))
        ea_arr = np.array([l[:min_len] for l in ea_layers])
        w_arr = np.array([l[:min_len] for l in w_layers])

        layer_ds = [d_eff(ea_arr[:, i], w_arr[:, i]) for i in range(min_len)]
        peak_l = int(np.argmax(np.abs(layer_ds)))
        peak_d = layer_ds[peak_l]
        ca_stats["per_layer"] = {
            "n_layers": min_len,
            "layer_ds": [float(d) for d in layer_ds],
            "peak_layer": peak_l,
            "peak_d": float(peak_d),
            "ea_n": len(ea_layers), "w_n": len(w_layers),
        }
        print(f"  Per-layer cross-attn: {min_len} layers, peak d={peak_d:+.3f} at L{peak_l}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(min_len)
        ea_m = ea_arr.mean(axis=0)
        w_m = w_arr.mean(axis=0)
        ea_se = ea_arr.std(axis=0) / np.sqrt(len(ea_arr))
        w_se = w_arr.std(axis=0) / np.sqrt(len(w_arr))
        ax.plot(x, ea_m, '-', color="#E74C3C", label=f"EA (n={len(ea_layers)})", lw=1.5)
        ax.fill_between(x, ea_m - 1.96*ea_se, ea_m + 1.96*ea_se, color="#E74C3C", alpha=0.1)
        ax.plot(x, w_m, '-', color="#3498DB", label=f"W (n={len(w_layers)})", lw=1.5)
        ax.fill_between(x, w_m - 1.96*w_se, w_m + 1.96*w_se, color="#3498DB", alpha=0.1)
        ax.set_xlabel("Cross-Attention Layer")
        ax.set_ylabel("Mean Entropy")
        ax.set_title(f"Llama Cross-Attention Entropy (peak d={peak_d:+.2f} at L{peak_l})",
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(FIGDIR / "fig_cross_attention.pdf")
        fig.savefig(FIGDIR / "fig_cross_attention.png")
        plt.close(fig)
        print("  Saved fig_cross_attention")

    all_stats["cross_attention"] = ca_stats


# ═══════════════════════════════════════════════════════════════
def analyze_encoding_robustness(qwen, llama):
    """Multi-prompt encoding cost robustness (ref0-3)."""
    print("\n" + "=" * 60)
    print("  ENCODING COST ROBUSTNESS (ref0-3)")
    print("=" * 60)

    rob_stats = {}
    for mname, recs in [("qwen", qwen), ("llama", llama)]:
        regs = by_r(recs)
        model_stats = {}
        for ref_idx in range(4):
            field = f"encoding_cost_ref{ref_idx}"
            ea_v = [r[field] for r in regs["EA"] if sf(r.get(field)) is not None]
            w_v = [r[field] for r in regs["W"] if sf(r.get(field)) is not None]
            if ea_v and w_v:
                d = d_eff(ea_v, w_v)
                _, p = stats.ttest_ind(ea_v, w_v)
                model_stats[f"ref{ref_idx}"] = {
                    "ea_mean": float(np.mean(ea_v)), "w_mean": float(np.mean(w_v)),
                    "ea_n": len(ea_v), "w_n": len(w_v),
                    "d": d, "p": float(p)
                }
                print(f"  [{mname}] ref{ref_idx}: EA={np.mean(ea_v):.3f}, W={np.mean(w_v):.3f}, "
                      f"d={d:+.3f}, p={p:.2e}")
        rob_stats[mname] = model_stats
    all_stats["encoding_robustness"] = rob_stats


# ═══════════════════════════════════════════════════════════════
def analyze_gradient_details(qwen):
    """Gradient attribution details (Qwen only)."""
    print("\n" + "=" * 60)
    print("  GRADIENT ATTRIBUTION DETAILS (Qwen)")
    print("=" * 60)

    regs = by_r(qwen)
    grad_stats = {}
    for field, label in [
        ("aesthetic_grad_norm_mean", "Grad norm mean"),
        ("aesthetic_grad_norm_std", "Grad norm std"),
        ("aesthetic_grad_norm_max", "Grad norm max"),
        ("aesthetic_grad_attribution_image_frac", "Image attribution frac"),
    ]:
        ea_v = [r[field] for r in regs["EA"] if sf(r.get(field)) is not None]
        w_v = [r[field] for r in regs["W"] if sf(r.get(field)) is not None]
        if ea_v and w_v:
            d = d_eff(ea_v, w_v)
            _, p = stats.ttest_ind(ea_v, w_v)
            grad_stats[field] = {
                "label": label,
                "ea_mean": float(np.mean(ea_v)), "w_mean": float(np.mean(w_v)),
                "ea_n": len(ea_v), "w_n": len(w_v),
                "d": d, "p": float(p)
            }
            print(f"  {label}: EA={np.mean(ea_v):.4f}, W={np.mean(w_v):.4f}, "
                  f"d={d:+.3f}, p={p:.2e}")
    all_stats["gradient_details"] = grad_stats


# ═══════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    qwen, llama = load_all()
    print(f"  Qwen: {len(qwen)} | Llama: {len(llama)}")

    analyze_temperature(qwen, llama)
    analyze_logit_distribution(qwen, llama)
    analyze_cross_attention(llama)
    analyze_encoding_robustness(qwen, llama)
    analyze_gradient_details(qwen)

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(STAT_FILE, 'w') as f:
        json.dump(all_stats, f, indent=2, default=convert)
    print(f"\nAll extended stats saved to {STAT_FILE}")
    print("=== EXTENDED ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
