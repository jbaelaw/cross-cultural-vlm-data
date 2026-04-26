#!/usr/bin/env python3
"""Generate publication figures for Scientific Reports manuscript.

This script reads the raw per-image result shards from
`results/runpod_final/`. Those shards are NOT redistributed in this GitHub
repository; they are deposited in the companion Zenodo archive. Pre-rendered
figure PDFs are already available in figures/.
"""

import json
import sys
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

if not (RESULTS / "runpod_final").exists():
    sys.exit(
        "Per-image result shards required. They are NOT in this GitHub "
        "repository. Download the Zenodo archive and place its "
        "derived_results/runpod_final/ directory at "
        f"{(RESULTS / 'runpod_final').relative_to(ROOT)}. "
        "Pre-rendered figures are in figures/."
    )

FIGDIR.mkdir(parents=True, exist_ok=True)

CULTURE_COLORS = {
    "korean": "#E74C3C",
    "chinese": "#E67E22",
    "japanese": "#2ECC71",
    "western": "#3498DB",
}
CULTURE_ORDER = ["korean", "chinese", "japanese", "western"]
CULTURE_LABELS = {"korean": "Korean", "chinese": "Chinese", "japanese": "Japanese", "western": "Western"}
REGION_COLORS = {"EA": "#E74C3C", "W": "#3498DB"}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def sf(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except:
        return None


def load_all():
    def _load(prefix, source):
        recs = []
        folder = "runpod_results" if source == "rp" else "gpu_raw"
        pattern = f"{prefix}_shard{{i}}/results.json" if source == "rp" else f"deep_{prefix}_shard{{i}}.json"
        for i in range(4):
            p = RESULTS / folder / pattern.format(i=i)
            try:
                with open(p) as f:
                    recs.extend(json.load(f))
            except:
                pass
        return recs

    qwen = _load("qwen3vl", "rp")
    llama_rp = _load("llama32v", "rp")
    llama_gpu = _load("llama32v", "gpu")
    llama = llama_gpu if len(llama_gpu) > len(llama_rp) else llama_rp

    with open(RESULTS / "compression_curated.json") as f:
        comp = {r["name"]: r for r in json.load(f)}
    return qwen, llama, comp


def vals(recs, field):
    return [v for v in (sf(r.get(field)) for r in recs) if v is not None]


def by_c(recs):
    return {c: [r for r in recs if r.get("culture") == c] for c in CULTURE_ORDER}


def by_r(recs):
    return {"EA": [r for r in recs if r.get("region") == "east_asian"],
            "W": [r for r in recs if r.get("region") == "western"]}


def d_eff(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    p = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / p if p > 1e-10 else 0.0


def figure1_scores(qwen, llama):
    """Figure 1: Aesthetic score distributions by culture, both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (model_name, recs) in zip(axes, [("Qwen3-VL-8B", qwen), ("Llama-3.2-Vision-11B", llama)]):
        groups = by_c(recs)
        data = [vals(groups[c], "aesthetic_score") for c in CULTURE_ORDER]

        bp = ax.boxplot(data, positions=range(4), widths=0.5, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='.', markersize=3, alpha=0.3),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))

        for patch, c in zip(bp['boxes'], CULTURE_ORDER):
            patch.set_facecolor(CULTURE_COLORS[c])
            patch.set_alpha(0.7)

        for i, c in enumerate(CULTURE_ORDER):
            v = vals(groups[c], "aesthetic_score")
            jitter = np.random.default_rng(42).normal(0, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v,
                       c=CULTURE_COLORS[c], alpha=0.08, s=8, zorder=0)

        ax.set_xticks(range(4))
        ax.set_xticklabels([CULTURE_LABELS[c] for c in CULTURE_ORDER], fontsize=10)
        ax.set_title(model_name, fontweight='bold')
        ax.set_ylabel('Aesthetic Score (1–10)' if ax == axes[0] else '')
        ax.set_ylim(1, 10.5)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ea_scores = []
        for c in ["korean", "chinese", "japanese"]:
            ea_scores.extend(vals(groups[c], "aesthetic_score"))
        w_scores = vals(groups["western"], "aesthetic_score")
        d = d_eff(ea_scores, w_scores)
        _, p = stats.ttest_ind(ea_scores, w_scores)
        pstr = f"p < 10$^{{{int(np.floor(np.log10(p)))}}}$" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.98, 0.02, f"EA vs W: d = {d:.2f}\n{pstr}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_scores.pdf")
    fig.savefig(FIGDIR / "fig1_scores.png")
    plt.close(fig)
    print(f"  Saved fig1_scores")


def figure2_probing(qwen, llama):
    """Figure 2: Internal probing signals — encoding cost, attention profile, effect summary."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 2a: Encoding cost by culture
    ax_a = fig.add_subplot(gs[0, 0])
    for model_name, recs, offset in [("Qwen", qwen, -0.15), ("Llama", llama, 0.15)]:
        groups = by_c(recs)
        means, cis = [], []
        for c in CULTURE_ORDER:
            v = vals(groups[c], "encoding_cost")
            if v:
                m = np.mean(v)
                se = stats.sem(v)
                means.append(m)
                cis.append(1.96 * se)
            else:
                means.append(0)
                cis.append(0)
        positions = np.arange(4) + offset
        bars = ax_a.bar(positions, means, 0.28, yerr=cis,
                        color=[CULTURE_COLORS[c] for c in CULTURE_ORDER],
                        alpha=0.7 if "Qwen" in model_name else 0.4,
                        edgecolor='black', linewidth=0.5,
                        capsize=3, label=model_name)
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels([CULTURE_LABELS[c] for c in CULTURE_ORDER])
    ax_a.set_ylabel('Encoding Cost (cross-entropy)')
    ax_a.set_title('(a) Encoding Cost by Culture', fontweight='bold', fontsize=11)
    ax_a.legend(framealpha=0.8, fontsize=9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(axis='y', alpha=0.3)

    # 2b: Attention entropy profile (Qwen only)
    ax_b = fig.add_subplot(gs[0, 1])
    regs = by_r(qwen)
    for region, label, color in [("EA", "East Asian", REGION_COLORS["EA"]),
                                  ("W", "Western", REGION_COLORS["W"])]:
        layers_data = []
        for r in regs[region]:
            ldata = r.get("aesthetic_attn_entropy_per_layer")
            if ldata and isinstance(ldata, list):
                layers_data.append(ldata)
        if layers_data:
            min_len = min(len(l) for l in layers_data)
            arr = np.array([l[:min_len] for l in layers_data])
            means = arr.mean(axis=0)
            sems = arr.std(axis=0) / np.sqrt(len(arr))
            x = np.arange(min_len)
            ax_b.plot(x, means, color=color, label=label, linewidth=1.5)
            ax_b.fill_between(x, means - 1.96 * sems, means + 1.96 * sems,
                              color=color, alpha=0.15)

    n_layers = min_len
    ax_b.axvspan(0, n_layers // 3, alpha=0.05, color='blue', label='_nolegend_')
    ax_b.axvspan(n_layers // 3, 2 * n_layers // 3, alpha=0.05, color='green', label='_nolegend_')
    ax_b.text(n_layers // 6, ax_b.get_ylim()[1] * 0.95, 'Early', ha='center', fontsize=8, alpha=0.5)
    ax_b.text(n_layers // 2, ax_b.get_ylim()[1] * 0.95, 'Mid', ha='center', fontsize=8, alpha=0.5)
    ax_b.text(5 * n_layers // 6, ax_b.get_ylim()[1] * 0.95, 'Late', ha='center', fontsize=8, alpha=0.5)

    ax_b.set_xlabel('Transformer Layer')
    ax_b.set_ylabel('Attention Entropy')
    ax_b.set_title('(b) Attention Entropy Profile (Qwen)', fontweight='bold', fontsize=11)
    ax_b.legend(framealpha=0.8, fontsize=9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(alpha=0.3)

    # 2c: Effect size summary (bottom, spanning full width)
    ax_c = fig.add_subplot(gs[1, :])
    signals = [
        ("Encoding cost", "encoding_cost"),
        ("Aesthetic PPL", "aesthetic_gen_perplexity"),
        ("Token entropy", "aesthetic_prob_entropy"),
        ("Hidden norm", "aesthetic_hidden_state_norm"),
        ("Top-1 confidence", "aesthetic_prob_top1_confidence"),
        ("Top-5 mass", "aesthetic_prob_top5_mass"),
        ("Prob. kurtosis", "aesthetic_prob_kurtosis"),
        ("Avg log-prob", "aesthetic_avg_logprob"),
    ]

    y_positions = np.arange(len(signals))
    for model_name, recs, marker, x_off in [("Qwen", qwen, 'o', -0.12), ("Llama", llama, 's', 0.12)]:
        regs_m = by_r(recs)
        ds = []
        for label, field in signals:
            ea_v = vals(regs_m["EA"], field)
            we_v = vals(regs_m["W"], field)
            if len(ea_v) >= 10 and len(we_v) >= 10:
                ds.append(d_eff(ea_v, we_v))
            else:
                ds.append(None)

        valid_y = [y + x_off for y, d in zip(y_positions, ds) if d is not None]
        valid_d = [d for d in ds if d is not None]
        colors = ['#E74C3C' if d > 0 else '#3498DB' for d in valid_d]
        ax_c.barh(valid_y, valid_d, height=0.22, color=colors, alpha=0.6 if marker == 'o' else 0.4,
                  edgecolor='black', linewidth=0.5, label=model_name)

    ax_c.axvline(0, color='black', linewidth=0.8)
    ax_c.axvline(0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_c.axvline(-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_c.axvline(0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_c.axvline(-0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels([s[0] for s in signals])
    ax_c.set_xlabel("Cohen's d (EA − W)")
    ax_c.set_title("(c) Effect Sizes: EA vs Western (model-agnostic signals)", fontweight='bold', fontsize=11)
    ax_c.legend(framealpha=0.8, fontsize=9, loc='lower right')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(axis='x', alpha=0.3)
    ax_c.text(0.22, -0.8, 'small', fontsize=7, alpha=0.4)
    ax_c.text(0.52, -0.8, 'medium', fontsize=7, alpha=0.4)
    ax_c.invert_yaxis()

    fig.savefig(FIGDIR / "fig2_probing.pdf")
    fig.savefig(FIGDIR / "fig2_probing.png")
    plt.close(fig)
    print(f"  Saved fig2_probing")


def figure3_compression(qwen, comp):
    """Figure 3: Compression–culture interaction."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    groups = by_c(qwen)

    # 3a: Spectral slope distributions
    ax = axes[0]
    for c in CULTURE_ORDER:
        v = [comp[r["name"]]["spectral_slope"] for r in groups[c]
             if r.get("name", "") in comp and "spectral_slope" in comp.get(r.get("name", ""), {})]
        if v:
            ax.hist(v, bins=30, alpha=0.5, color=CULTURE_COLORS[c],
                    label=f"{CULTURE_LABELS[c]} (n={len(v)})", density=True, edgecolor='none')
            ax.axvline(np.mean(v), color=CULTURE_COLORS[c], linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Spectral Slope')
    ax.set_ylabel('Density')
    ax.set_title('(a) Spectral Slope by Culture\n(d = −1.72, EA vs W)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3b: Within-culture correlation heatmap
    ax = axes[1]
    comp_fields = ["spectral_slope", "gzip_ratio", "block_entropy", "pixel_entropy"]
    vlm_fields = ["aesthetic_score", "encoding_cost", "aesthetic_gen_perplexity"]
    comp_labels = ["Spec. slope", "Gzip ratio", "Block ent.", "Pixel ent."]
    vlm_labels = ["Aesth. score", "Enc. cost", "Aesth. PPL"]

    corr_by_culture = {}
    for c in CULTURE_ORDER:
        corr_matrix = np.zeros((len(comp_fields), len(vlm_fields)))
        for i, cf in enumerate(comp_fields):
            for j, vf in enumerate(vlm_fields):
                xs, ys = [], []
                for r in groups[c]:
                    y = sf(r.get(vf))
                    cm = comp.get(r.get("name", ""))
                    if y is not None and cm and cf in cm:
                        xs.append(cm[cf])
                        ys.append(y)
                if len(xs) >= 10:
                    corr_matrix[i, j] = stats.spearmanr(xs, ys).statistic
        corr_by_culture[c] = corr_matrix

    avg_corr = np.mean([corr_by_culture[c] for c in CULTURE_ORDER], axis=0)
    std_corr = np.std([corr_by_culture[c] for c in CULTURE_ORDER], axis=0)

    im = ax.imshow(std_corr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.2)
    ax.set_xticks(range(len(vlm_fields)))
    ax.set_xticklabels(vlm_labels, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(comp_fields)))
    ax.set_yticklabels(comp_labels, fontsize=8)
    for i in range(len(comp_fields)):
        for j in range(len(vlm_fields)):
            ax.text(j, i, f"{std_corr[i, j]:.2f}", ha='center', va='center', fontsize=7.5,
                    color='white' if std_corr[i, j] > 0.1 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cross-culture SD(ρ)')
    ax.set_title('(b) Correlation Variability\nAcross Cultures', fontweight='bold', fontsize=10)

    # 3c: Korean gzip anomaly
    ax = axes[2]
    gzip_by_c = {}
    for c in CULTURE_ORDER:
        v = [comp[r["name"]]["gzip_ratio"] for r in groups[c]
             if r.get("name", "") in comp and "gzip_ratio" in comp.get(r.get("name", ""), {})]
        gzip_by_c[c] = v

    bp = ax.boxplot([gzip_by_c[c] for c in CULTURE_ORDER],
                    positions=range(4), widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='.', markersize=3, alpha=0.3))
    for patch, c in zip(bp['boxes'], CULTURE_ORDER):
        patch.set_facecolor(CULTURE_COLORS[c])
        patch.set_alpha(0.7)

    ax.set_xticks(range(4))
    ax.set_xticklabels([CULTURE_LABELS[c] for c in CULTURE_ORDER], fontsize=9)
    ax.set_ylabel('Gzip Compression Ratio')
    ax.set_title('(c) Korean Gzip Anomaly\n(d = +1.00 vs CN+JP)', fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    kr = gzip_by_c["korean"]
    cj = gzip_by_c["chinese"] + gzip_by_c["japanese"]
    d = d_eff(np.array(kr), np.array(cj))
    ax.annotate(f'd = +{d:.2f}***\nvs CN+JP', xy=(0, np.mean(kr)),
                xytext=(2.5, np.mean(kr) + 0.05),
                fontsize=8, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_compression.pdf")
    fig.savefig(FIGDIR / "fig3_compression.png")
    plt.close(fig)
    print(f"  Saved fig3_compression")


def main():
    print("Loading data...")
    qwen, llama, comp = load_all()
    print(f"  Qwen: {len(qwen)} | Llama: {len(llama)} | Compression: {len(comp)}")

    print("\nGenerating figures...")
    figure1_scores(qwen, llama)
    figure2_probing(qwen, llama)
    figure3_compression(qwen, comp)
    print(f"\nAll figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
