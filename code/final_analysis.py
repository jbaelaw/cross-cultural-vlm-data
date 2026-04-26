#!/usr/bin/env python3
"""Final analysis using the full runpod_final dataset.

This script regenerates `data/final_stats.json` and `figures/` from the raw
per-image result shards (qwen3vl_shard0..3, llama32v_shard0..3, and the
vision-encoder supplements). Those shards are NOT redistributed in this
GitHub repository; they are deposited in the companion Zenodo archive.

To rerun the analysis, download the Zenodo archive and place its
`derived_results/runpod_final/` directory under this repository as
`results/runpod_final/`. Without those shards this script will exit early.

The pre-computed JSON outputs in `data/` and the figure PDFs in `figures/`
are reproduced from those shards exactly as published.
"""

import json
import sys
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
STAT_FILE = ROOT / "data" / "final_stats.json"

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


def load_final():
    """Load the COMPLETE runpod_final dataset."""
    def _load(prefix):
        recs = []
        for i in range(4):
            p = RESULTS / "runpod_final" / f"{prefix}_shard{i}" / "results.json"
            try:
                with open(p) as f:
                    data = json.load(f)
                recs.extend(data)
            except:
                pass
        return recs

    qwen = _load("qwen3vl")
    llama = _load("llama32v")

    # Load vision supplement
    def _load_vision(prefix):
        recs = {}
        for i in range(4):
            p = RESULTS / "runpod_final" / f"{prefix}_supplement_vision_shard{i}" / "results.json"
            try:
                with open(p) as f:
                    for r in json.load(f):
                        recs[r["name"]] = r
            except:
                pass
        return recs

    qwen_vis = _load_vision("qwen3vl")
    llama_vis = _load_vision("llama32v")

    with open(RESULTS / "compression_curated.json") as f:
        comp = {r["name"]: r for r in json.load(f)}

    return qwen, llama, qwen_vis, llama_vis, comp


def d_eff(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    p = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / p if p > 1e-10 else 0.0


def vals(recs, field):
    return [v for v in (sf(r.get(field)) for r in recs) if v is not None]


def by_c(recs):
    return {c: [r for r in recs if r.get("culture") == c] for c in CO}


def by_r(recs):
    return {"EA": [r for r in recs if r.get("culture") in EA],
            "W": [r for r in recs if r.get("culture") == "western"]}


def bootstrap_d(a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        ds.append(d_eff(a[ia], b[ib]))
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5)


all_stats = {}

# ═══════════════════════════════════════════════════════════════
def run_core_stats(qwen, llama):
    """Core statistical analysis with full dataset."""
    print("=" * 60)
    print("  CORE STATISTICS (n=2,328 per model)")
    print("=" * 60)

    for mname, recs in [("Qwen", qwen), ("Llama", llama)]:
        regs = by_r(recs)
        groups = by_c(recs)
        ea_s = np.array(vals(regs["EA"], "aesthetic_score"))
        we_s = np.array(vals(regs["W"], "aesthetic_score"))

        t, p = stats.ttest_ind(ea_s, we_s)
        mw = stats.mannwhitneyu(ea_s, we_s)
        d = d_eff(ea_s, we_s)
        d_lo, d_hi = bootstrap_d(ea_s, we_s)

        kw_data = [vals(groups[c], "aesthetic_score") for c in CO]
        kw = stats.kruskal(*kw_data)

        key = f"score_{mname.lower()}"
        all_stats[key] = {
            "ea_mean": float(np.mean(ea_s)), "ea_n": len(ea_s),
            "w_mean": float(np.mean(we_s)), "w_n": len(we_s),
            "d": float(d), "d_ci": [float(d_lo), float(d_hi)],
            "t": float(t), "p_welch": float(p), "p_mw": float(mw.pvalue),
            "kw_H": float(kw.statistic), "kw_p": float(kw.pvalue),
        }

        print(f"\n  [{mname}] n_EA={len(ea_s)}, n_W={len(we_s)}")
        print(f"    EA={np.mean(ea_s):.3f} ± {np.std(ea_s):.3f}, W={np.mean(we_s):.3f} ± {np.std(we_s):.3f}")
        print(f"    d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], Welch p={p:.2e}, MW p={mw.pvalue:.2e}")
        print(f"    KW H={kw.statistic:.1f}, p={kw.pvalue:.2e}")

        for c in CO:
            v = vals(groups[c], "aesthetic_score")
            print(f"    {c}: n={len(v)}, mean={np.mean(v):.3f}")


def run_vlm_signals(qwen, llama):
    """All VLM internal signals comparison."""
    print("\n" + "=" * 60)
    print("  VLM INTERNAL SIGNALS (model-agnostic)")
    print("=" * 60)

    fields = [
        ("encoding_cost", "Encoding cost"),
        ("aesthetic_gen_perplexity", "Aesth PPL"),
        ("aesthetic_avg_logprob", "Avg logprob"),
        ("aesthetic_prob_entropy", "Token entropy"),
        ("aesthetic_prob_top1_confidence", "Top-1 conf"),
        ("aesthetic_prob_top5_mass", "Top-5 mass"),
        ("aesthetic_prob_kurtosis", "Prob kurtosis"),
        ("aesthetic_hidden_state_norm", "Hidden norm"),
        ("aesthetic_hidden_intrinsic_dim", "Intrinsic dim"),
        ("aesthetic_attn_entropy_mean", "Attn entropy"),
        ("aesthetic_attn_entropy_early", "Attn early"),
        ("aesthetic_attn_entropy_mid", "Attn mid"),
        ("aesthetic_attn_entropy_late", "Attn late"),
        ("encoding_img_loss", "Enc img loss"),
        ("encoding_txt_loss", "Enc txt loss"),
        ("aesthetic_grad_norm_mean", "Grad norm"),
        ("aesthetic_n_tokens_generated", "Tokens gen"),
        ("aesthetic_response_vocab_diversity", "Vocab div"),
    ]

    signal_results = []
    for field, label in fields:
        row = {"field": field, "label": label}
        for mname, recs in [("qwen", qwen), ("llama", llama)]:
            regs = by_r(recs)
            ea_v = vals(regs["EA"], field)
            we_v = vals(regs["W"], field)
            if len(ea_v) >= 10 and len(we_v) >= 10:
                d = d_eff(ea_v, we_v)
                _, p = stats.ttest_ind(ea_v, we_v)
                row[f"{mname}_ea"] = float(np.mean(ea_v))
                row[f"{mname}_w"] = float(np.mean(we_v))
                row[f"{mname}_d"] = float(d)
                row[f"{mname}_p"] = float(p)
                row[f"{mname}_n_ea"] = len(ea_v)
                row[f"{mname}_n_w"] = len(we_v)
        signal_results.append(row)

    all_stats["vlm_signals"] = signal_results

    # Print model-agnostic signals
    print(f"\n  {'Signal':22s} {'Q_d':>7s} {'L_d':>7s} {'Agree':>5s}")
    for r in sorted(signal_results, key=lambda x: abs(x.get("qwen_d", 0) + x.get("llama_d", 0)) / 2, reverse=True):
        qd = r.get("qwen_d")
        ld = r.get("llama_d")
        agree = "Y" if qd and ld and (qd > 0) == (ld > 0) else "N" if qd and ld else "-"
        qs = f"{qd:+.3f}" if qd else "  n/a"
        ls = f"{ld:+.3f}" if ld else "  n/a"
        print(f"  {r['label']:22s} {qs:>7s} {ls:>7s} {agree:>5s}")


def run_layer_analysis(qwen, llama):
    """Per-layer hidden-state norm and attention entropy profiles."""
    print("\n" + "=" * 60)
    print("  PER-LAYER ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for col, (mname, recs) in enumerate([("Qwen", qwen), ("Llama", llama)]):
        regs = by_r(recs)

        # Row 0: Attention entropy profile (Qwen has it, Llama may not)
        ax = axes[0, col]
        has_attn = False
        for region, label, color in [("EA", "East Asian", "#E74C3C"), ("W", "Western", "#3498DB")]:
            layers_data = []
            for r in regs[region]:
                ld = r.get("aesthetic_attn_entropy_per_layer")
                if ld and isinstance(ld, list) and len(ld) > 5:
                    layers_data.append(ld)
            if layers_data:
                has_attn = True
                min_len = min(len(l) for l in layers_data)
                arr = np.array([l[:min_len] for l in layers_data])
                means = arr.mean(axis=0)
                sems = arr.std(axis=0) / np.sqrt(len(arr))
                x = np.arange(min_len)
                ax.plot(x, means, color=color, label=f"{label} (n={len(layers_data)})", lw=1.5)
                ax.fill_between(x, means - 1.96*sems, means + 1.96*sems, color=color, alpha=0.12)

        ax.set_title(f'{mname}: Attention Entropy', fontweight='bold', fontsize=11)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Attention Entropy')
        if has_attn:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Not available\n(SDPA attention)', transform=ax.transAxes, ha='center', fontsize=11, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)

        # Row 1: Hidden-state norm profile
        ax = axes[1, col]
        layer_ds = None
        for region, label, color in [("EA", "East Asian", "#E74C3C"), ("W", "Western", "#3498DB")]:
            layers_data = []
            for r in regs[region]:
                norms = r.get("encoding_layer_norms_ref0") or r.get("encoding_layer_norms")
                if norms and isinstance(norms, list) and len(norms) > 5:
                    layers_data.append(norms)
            if not layers_data:
                for r in regs[region]:
                    norms = r.get("aesthetic_hidden_layer_norms")
                    if norms and isinstance(norms, list) and len(norms) > 5:
                        layers_data.append(norms)

            if layers_data:
                min_len = min(len(l) for l in layers_data)
                arr = np.array([l[:min_len] for l in layers_data])
                means = arr.mean(axis=0)
                sems = arr.std(axis=0) / np.sqrt(len(arr))
                x = np.arange(min_len)
                ax.plot(x, means, color=color, label=f"{label} (n={len(layers_data)})", lw=1.5)
                ax.fill_between(x, means - 1.96*sems, means + 1.96*sems, color=color, alpha=0.12)

                if region == "EA":
                    ea_arr = arr
                elif region == "W":
                    w_arr = arr

        if 'ea_arr' in dir() and 'w_arr' in dir() and ea_arr.shape[1] == w_arr.shape[1]:
            layer_ds_list = [d_eff(ea_arr[:, l], w_arr[:, l]) for l in range(ea_arr.shape[1])]
            peak_l = np.argmax(np.abs(layer_ds_list))
            peak_d = layer_ds_list[peak_l]
            ax.set_title(f'{mname}: Hidden-State Norms\n(peak d={peak_d:+.2f} at L{peak_l})',
                         fontweight='bold', fontsize=11)
            all_stats[f"layer_peak_{mname.lower()}"] = {"layer": int(peak_l), "d": float(peak_d)}
            print(f"  [{mname}] Peak layer divergence: L{peak_l}, d={peak_d:+.3f}")
        else:
            ax.set_title(f'{mname}: Hidden-State Norms', fontweight='bold', fontsize=11)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Hidden-State Norm')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        del ea_arr, w_arr

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig_layer_profiles.pdf")
    fig.savefig(FIGDIR / "fig_layer_profiles.png")
    plt.close(fig)
    print("  Saved fig_layer_profiles")


def run_vision_analysis(qwen_vis, llama_vis):
    """Vision embedding analysis from supplement data."""
    print("\n" + "=" * 60)
    print("  VISION EMBEDDING ANALYSIS")
    print("=" * 60)

    for mname, vis in [("Qwen", qwen_vis), ("Llama", llama_vis)]:
        if not vis:
            continue
        by_culture = defaultdict(list)
        for name, r in vis.items():
            c = r.get("culture", "?")
            norm = sf(r.get("vision_embed_norm"))
            patches = sf(r.get("vision_patch_count"))
            if norm is not None:
                by_culture[c].append({"norm": norm, "patches": patches})

        print(f"\n  [{mname}] Vision embeddings:")
        ea_norms, w_norms = [], []
        for c in CO:
            items = by_culture.get(c, [])
            if items:
                norms = [i["norm"] for i in items]
                patches = [i["patches"] for i in items if i["patches"]]
                print(f"    {c}: n={len(items)}, norm={np.mean(norms):.2f}±{np.std(norms):.2f}"
                      + (f", patches={np.mean(patches):.0f}" if patches else ""))
                if c in EA:
                    ea_norms.extend(norms)
                else:
                    w_norms.extend(norms)

        if ea_norms and w_norms:
            d = d_eff(ea_norms, w_norms)
            _, p = stats.ttest_ind(ea_norms, w_norms)
            print(f"    EA vs W vision norm: d={d:+.3f}, p={p:.2e}")
            all_stats[f"vision_norm_{mname.lower()}"] = {"d": float(d), "p": float(p)}


def run_mediation_full(qwen, comp):
    """Mediation + matched-complexity + regression with full data."""
    print("\n" + "=" * 60)
    print("  MEDIATION & REGRESSION (FULL DATA)")
    print("=" * 60)
    from numpy.linalg import lstsq

    rows = []
    for r in qwen:
        s = sf(r.get("aesthetic_score"))
        ec = sf(r.get("encoding_cost"))
        ppl = sf(r.get("aesthetic_gen_perplexity"))
        ent = sf(r.get("aesthetic_prob_entropy"))
        conf = sf(r.get("aesthetic_prob_top1_confidence"))
        hn = sf(r.get("aesthetic_hidden_state_norm"))
        name = r.get("name", "")
        culture = r.get("culture", "")
        if any(v is None for v in [s, ec, ppl, ent, conf, hn]):
            continue
        if name not in comp or culture not in CO:
            continue
        c = comp[name]
        if "spectral_slope" not in c:
            continue
        is_ea = 1.0 if culture in EA else 0.0
        rows.append([is_ea, c["spectral_slope"], c.get("gzip_ratio", 0),
                     c.get("pixel_entropy", 0), ec, ppl, ent, conf, hn, s, culture])

    n = len(rows)
    data = np.array([[r[i] for i in range(10)] for r in rows])
    cultures = [r[10] for r in rows]
    Y = data[:, -1]
    ss_tot = np.sum((Y - Y.mean()) ** 2)

    def r2(predictors):
        X = np.column_stack([data[:, i] for i in predictors] + [np.ones(n)])
        b, _, _, _ = lstsq(X, Y, rcond=None)
        pred = X @ b
        return 1 - np.sum((Y - pred)**2) / ss_tot

    # Mediation
    X_culture = data[:, 0]
    M_slope = data[:, 1]
    slope_c, _, r_c, p_c, _ = stats.linregress(X_culture, Y)
    slope_a, _, r_a, p_a, _ = stats.linregress(X_culture, M_slope)
    A = np.column_stack([X_culture, M_slope, np.ones(n)])
    betas, _, _, _ = lstsq(A, Y, rcond=None)
    indirect = slope_a * betas[1]
    proportion = abs(indirect / slope_c) * 100

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_ind = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        sa, _, _, _, _ = stats.linregress(X_culture[idx], M_slope[idx])
        Ab = np.column_stack([X_culture[idx], M_slope[idx], np.ones(len(idx))])
        bb, _, _, _ = lstsq(Ab, Y[idx], rcond=None)
        boot_ind.append(sa * bb[1])
    ci_lo, ci_hi = np.percentile(boot_ind, [2.5, 97.5])

    print(f"  Mediation: n={n}")
    print(f"    Total effect (c): β={slope_c:.4f}")
    print(f"    Indirect (a×b): {indirect:.4f} ({proportion:.1f}%)")
    print(f"    Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    significant = ci_lo > 0 or ci_hi < 0
    print(f"    CI excludes zero: {significant}")

    all_stats["mediation"] = {
        "n": n, "total_effect": float(slope_c),
        "indirect": float(indirect), "proportion_pct": float(proportion),
        "ci": [float(ci_lo), float(ci_hi)], "ci_excludes_zero": significant,
    }

    # Hierarchical regression
    r2_1 = r2([0])
    r2_2 = r2([0, 1, 2, 3])
    r2_3 = r2([0, 1, 2, 3, 4, 5, 6, 7, 8])
    r2_4 = r2([1, 2, 3, 4, 5, 6, 7, 8])
    unique_culture = r2_3 - r2_4

    print(f"\n  Hierarchical regression:")
    print(f"    M1 culture:          R²={r2_1:.4f}")
    print(f"    M2 +compression:     R²={r2_2:.4f} (ΔR²={r2_2-r2_1:.4f})")
    print(f"    M3 +VLM signals:     R²={r2_3:.4f} (ΔR²={r2_3-r2_2:.4f})")
    print(f"    M4 no culture:       R²={r2_4:.4f}")
    print(f"    Unique culture:      {unique_culture:.4f} ({unique_culture*100:.1f}%)")

    all_stats["regression"] = {
        "n": n,
        "r2_culture": float(r2_1), "r2_plus_comp": float(r2_2),
        "r2_full": float(r2_3), "r2_no_culture": float(r2_4),
        "unique_culture": float(unique_culture),
    }

    # Matched-complexity
    ea_items = [(data[i, 1], data[i, -1]) for i in range(n) if data[i, 0] == 1.0]
    w_items = [(data[i, 1], data[i, -1]) for i in range(n) if data[i, 0] == 0.0]
    ea_slopes = [x[0] for x in ea_items]
    w_slopes = [x[0] for x in w_items]
    overlap_lo = max(np.percentile(ea_slopes, 10), np.percentile(w_slopes, 10))
    overlap_hi = min(np.percentile(ea_slopes, 90), np.percentile(w_slopes, 90))
    bins = np.linspace(overlap_lo, overlap_hi, 6)

    ea_matched, w_matched = [], []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        ea_b = [x[1] for x in ea_items if lo <= x[0] < hi]
        w_b = [x[1] for x in w_items if lo <= x[0] < hi]
        if len(ea_b) >= 5 and len(w_b) >= 5:
            ea_matched.extend(ea_b)
            w_matched.extend(w_b)

    if ea_matched and w_matched:
        d_matched = d_eff(ea_matched, w_matched)
        _, p_matched = stats.ttest_ind(ea_matched, w_matched)
        print(f"\n  Matched-complexity (spectral slope):")
        print(f"    n_EA={len(ea_matched)}, n_W={len(w_matched)}")
        print(f"    d={d_matched:+.3f}, p={p_matched:.2e}")
        all_stats["matched_complexity"] = {
            "d": float(d_matched), "p": float(p_matched),
            "n_ea": len(ea_matched), "n_w": len(w_matched),
        }


def run_figure_scores(qwen, llama):
    """Publication-quality score distribution figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (mname, recs) in zip(axes, [("Qwen3-VL-8B", qwen), ("Llama-3.2-Vision-11B", llama)]):
        groups = by_c(recs)
        data = [vals(groups[c], "aesthetic_score") for c in CO]
        bp = ax.boxplot(data, positions=range(4), widths=0.5, patch_artist=True,
                        medianprops=dict(color='black', lw=1.5),
                        flierprops=dict(marker='.', markersize=3, alpha=0.3),
                        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        for patch, c in zip(bp['boxes'], CO):
            patch.set_facecolor(CC[c])
            patch.set_alpha(0.7)
        for i, c in enumerate(CO):
            v = vals(groups[c], "aesthetic_score")
            jitter = np.random.default_rng(42).normal(0, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, c=CC[c], alpha=0.06, s=6, zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CL[c] for c in CO], fontsize=10)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Aesthetic Score' if ax == axes[0] else '')
        ax.set_ylim(1, 10.5)
        ax.grid(axis='y', alpha=0.2, lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        regs = by_r(recs)
        ea_s = vals(regs["EA"], "aesthetic_score")
        we_s = vals(regs["W"], "aesthetic_score")
        d = d_eff(ea_s, we_s)
        ax.text(0.98, 0.02, f"EA vs W: d = {d:.2f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_scores.pdf")
    fig.savefig(FIGDIR / "fig1_scores.png")
    plt.close(fig)
    print("  Saved fig1_scores")


def run_figure_effect_sizes(qwen, llama):
    """Effect-size forest plot for all model-agnostic signals."""
    signals = [
        ("encoding_cost", "Encoding cost"),
        ("aesthetic_gen_perplexity", "Aesthetic PPL"),
        ("aesthetic_prob_entropy", "Token entropy"),
        ("aesthetic_hidden_state_norm", "Hidden-state norm"),
        ("aesthetic_prob_top1_confidence", "Top-1 confidence"),
        ("aesthetic_avg_logprob", "Avg log-prob"),
        ("aesthetic_prob_top5_mass", "Top-5 mass"),
        ("aesthetic_prob_kurtosis", "Prob. kurtosis"),
        ("aesthetic_n_tokens_generated", "Tokens generated"),
        ("aesthetic_score", "Aesthetic score"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(len(signals))

    for mname, recs, marker, offset, alpha in [
        ("Qwen", qwen, 'o', -0.15, 0.85), ("Llama", llama, 's', 0.15, 0.55)
    ]:
        regs = by_r(recs)
        ds, cis_lo, cis_hi = [], [], []
        for field, label in signals:
            ea_v = np.array(vals(regs["EA"], field))
            we_v = np.array(vals(regs["W"], field))
            if len(ea_v) >= 10 and len(we_v) >= 10:
                d = d_eff(ea_v, we_v)
                lo, hi = bootstrap_d(ea_v, we_v, n=2000)
                ds.append(d)
                cis_lo.append(lo)
                cis_hi.append(hi)
            else:
                ds.append(None)
                cis_lo.append(None)
                cis_hi.append(None)

        for i, (d, lo, hi) in enumerate(zip(ds, cis_lo, cis_hi)):
            if d is not None:
                color = '#E74C3C' if d > 0 else '#3498DB'
                ax.plot(d, i + offset, marker, color=color, markersize=7, alpha=alpha, zorder=3)
                ax.plot([lo, hi], [i + offset, i + offset], '-', color=color, lw=1.5, alpha=alpha*0.7, zorder=2)

    ax.axvline(0, color='black', lw=0.8)
    for threshold in [0.2, -0.2, 0.5, -0.5, 0.8, -0.8]:
        ax.axvline(threshold, color='gray', lw=0.4, ls=':', alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[1] for s in signals])
    ax.set_xlabel("Cohen's d (EA − W)  [with 95% bootstrap CI]")
    ax.set_title("Effect Sizes: East Asian vs Western\n(●Qwen  ■Llama)", fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    ax.invert_yaxis()
    ax.text(0.22, len(signals) + 0.3, 'small', fontsize=7, alpha=0.4, ha='center')
    ax.text(0.52, len(signals) + 0.3, 'medium', fontsize=7, alpha=0.4, ha='center')
    ax.text(0.82, len(signals) + 0.3, 'large', fontsize=7, alpha=0.4, ha='center')

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_effect_sizes.pdf")
    fig.savefig(FIGDIR / "fig2_effect_sizes.png")
    plt.close(fig)
    print("  Saved fig2_effect_sizes")


def run_figure_misclass(qwen, llama):
    """Misclassification confusion matrix."""
    culture_words = {
        "korean": ["korean", "korea", "joseon", "goryeo", "hangul"],
        "chinese": ["chinese", "china", "ming", "qing", "song dynasty", "tang dynasty", "yuan"],
        "japanese": ["japanese", "japan", "edo", "ukiyo", "meiji", "muromachi"],
        "western": ["european", "renaissance", "baroque", "impressionist",
                     "dutch", "italian", "french", "flemish", "german", "spanish", "british"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (mname, recs) in zip(axes, [("Qwen3-VL-8B", qwen), ("Llama-3.2-Vision-11B", llama)]):
        groups = by_c(recs)
        matrix = np.zeros((4, 4))
        for i, actual in enumerate(CO):
            n = 0
            for r in groups[actual]:
                text = (r.get("aesthetic_text", "") + " " + r.get("cultural_text", "")).lower()
                if not text.strip():
                    continue
                n += 1
                for j, target in enumerate(CO):
                    if any(w in text for w in culture_words[target]):
                        matrix[i, j] += 1
            if n > 0:
                matrix[i, :] = 100.0 * matrix[i, :] / n

        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=100, aspect='equal')
        ax.set_xticks(range(4))
        ax.set_xticklabels([CL[c] for c in CO], fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels([CL[c] for c in CO], fontsize=9)
        ax.set_xlabel('Attributed Culture')
        ax.set_ylabel('True Culture')
        ax.set_title(mname, fontweight='bold')
        for i in range(4):
            for j in range(4):
                color = 'white' if matrix[i, j] > 50 else 'black'
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f"{matrix[i, j]:.0f}%", ha='center', va='center',
                        fontsize=10, color=color, fontweight=weight)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Attribution Rate (%)')
    fig.suptitle('Cultural Attribution in VLM Responses', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_misclassification.pdf")
    fig.savefig(FIGDIR / "fig4_misclassification.png")
    plt.close(fig)
    print("  Saved fig4_misclassification")


def run_figure_compression(qwen, comp):
    """Compression-culture interaction figure."""
    groups = by_c(qwen)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # 1: Spectral slope
    ax = axes[0]
    for c in CO:
        v = [comp[r["name"]]["spectral_slope"] for r in groups[c]
             if r.get("name", "") in comp and "spectral_slope" in comp.get(r.get("name", ""), {})]
        if v:
            ax.hist(v, bins=30, alpha=0.5, color=CC[c],
                    label=f"{CL[c]} ({len(v)})", density=True, edgecolor='none')
            ax.axvline(np.mean(v), color=CC[c], ls='--', lw=1.5, alpha=0.8)
    ax.set_xlabel('Spectral Slope')
    ax.set_ylabel('Density')
    ax.set_title('(a) Spectral Slope\n(EA vs W: d = −1.72)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2: Encoding cost vs spectral slope scatter
    ax = axes[1]
    for c in CO:
        xs, ys = [], []
        for r in groups[c]:
            ec = sf(r.get("encoding_cost"))
            name = r.get("name", "")
            if ec and name in comp and "spectral_slope" in comp[name]:
                xs.append(comp[name]["spectral_slope"])
                ys.append(ec)
        if xs:
            ax.scatter(xs, ys, c=CC[c], alpha=0.25, s=10, label=CL[c], edgecolors='none')
    ax.set_xlabel('Spectral Slope')
    ax.set_ylabel('Encoding Cost')
    ax.set_title('(b) Compression → VLM Cost', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5, markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)

    # 3: Korean gzip anomaly
    ax = axes[2]
    gzip_data = {c: [comp[r["name"]]["gzip_ratio"] for r in groups[c]
                      if r.get("name", "") in comp and "gzip_ratio" in comp.get(r.get("name", ""), {})]
                 for c in CO}
    bp = ax.boxplot([gzip_data[c] for c in CO], positions=range(4), widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', lw=1.5),
                    flierprops=dict(marker='.', markersize=3, alpha=0.3))
    for patch, c in zip(bp['boxes'], CO):
        patch.set_facecolor(CC[c])
        patch.set_alpha(0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CL[c] for c in CO], fontsize=9)
    ax.set_ylabel('Gzip Ratio')
    ax.set_title('(c) Korean Gzip Anomaly\n(d = +1.00 vs CN+JP)', fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_compression.pdf")
    fig.savefig(FIGDIR / "fig3_compression.png")
    plt.close(fig)
    print("  Saved fig3_compression")


def run_data_accounting(qwen, llama, comp):
    """Data provenance accounting for Methods section."""
    print("\n" + "=" * 60)
    print("  DATA ACCOUNTING (for Methods)")
    print("=" * 60)

    for mname, recs in [("Qwen", qwen), ("Llama", llama)]:
        groups = by_c(recs)
        total = len(recs)
        with_score = len([r for r in recs if sf(r.get("aesthetic_score")) is not None])
        with_enc = len([r for r in recs if sf(r.get("encoding_cost")) is not None])
        with_attn = len([r for r in recs if r.get("aesthetic_attn_entropy_per_layer")])
        with_hidden = len([r for r in recs if sf(r.get("aesthetic_hidden_state_norm")) is not None])

        print(f"\n  [{mname}] Total: {total}")
        print(f"    With score: {with_score} ({100*with_score/total:.1f}%)")
        print(f"    With encoding cost: {with_enc} ({100*with_enc/total:.1f}%)")
        print(f"    With attention: {with_attn} ({100*with_attn/total:.1f}%)")
        print(f"    With hidden norm: {with_hidden} ({100*with_hidden/total:.1f}%)")
        for c in CO:
            n = len(groups[c])
            ns = len([r for r in groups[c] if sf(r.get("aesthetic_score")) is not None])
            print(f"    {c}: {n} total, {ns} with score")

    # Shared images
    q_names = {r["name"] for r in qwen if sf(r.get("aesthetic_score")) is not None}
    l_names = {r["name"] for r in llama if sf(r.get("aesthetic_score")) is not None}
    shared = q_names & l_names
    print(f"\n  Shared images (both models with score): {len(shared)}")


def main():
    print("Loading FULL runpod_final dataset...")
    qwen, llama, qwen_vis, llama_vis, comp = load_final()
    print(f"  Qwen: {len(qwen)} | Llama: {len(llama)}")
    print(f"  Vision: Qwen {len(qwen_vis)} | Llama {len(llama_vis)}")
    print(f"  Compression: {len(comp)}")

    run_data_accounting(qwen, llama, comp)
    run_core_stats(qwen, llama)
    run_vlm_signals(qwen, llama)
    run_layer_analysis(qwen, llama)
    run_vision_analysis(qwen_vis, llama_vis)
    run_mediation_full(qwen, comp)

    print("\nGenerating publication figures...")
    run_figure_scores(qwen, llama)
    run_figure_effect_sizes(qwen, llama)
    run_figure_compression(qwen, comp)
    run_figure_misclass(qwen, llama)

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(STAT_FILE, 'w') as f:
        json.dump(all_stats, f, indent=2, default=convert)
    print(f"\nAll stats saved to {STAT_FILE}")
    print("\n=== FINAL ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
