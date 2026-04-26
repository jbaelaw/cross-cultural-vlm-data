#!/usr/bin/env python3
"""Complete verification: ALL manuscript numbers vs. source data."""

import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
with open(DATA / "final_stats.json") as f:
    F = json.load(f)
with open(DATA / "extended_stats.json") as f:
    E = json.load(f)

errors = []
checks = 0

def c(desc, mval, jval, tol=0.015):
    global checks; checks += 1
    try:
        m, j = float(mval), float(jval)
        if abs(m - j) > tol:
            errors.append(f"  FAIL {desc}: manu={m}, json={j}, Δ={m-j:+.4f}")
    except Exception as e:
        errors.append(f"  ERR  {desc}: '{mval}' vs '{jval}' ({e})")

def pb(desc, exp, pval):
    global checks; checks += 1
    bound = 10 ** exp
    if pval > bound:
        errors.append(f"  FAIL {desc}: p={pval:.2e} > 10^{exp}={bound:.2e}")

print("=" * 70)
print("  PART A: ORIGINAL MANUSCRIPT NUMBERS (from final_stats.json)")
print("=" * 70)

# Score comparisons
c("Qwen EA score", 6.94, F["score_qwen"]["ea_mean"], 0.005)
c("Qwen W score", 7.23, F["score_qwen"]["w_mean"], 0.005)
c("Qwen score d", -0.46, F["score_qwen"]["d"])
c("Llama EA score", 7.57, F["score_llama"]["ea_mean"], 0.01)  # ms says ~7.56-7.57 region
c("Llama W score", 7.90, F["score_llama"]["w_mean"], 0.01)
c("Llama score d", -0.36, F["score_llama"]["d"])

# VLM signals (spot-check the key ones)
vs = F.get("vlm_signals", [])
for sig in vs:
    if isinstance(sig, dict) and "qwen_d" in sig and "llama_d" in sig:
        print(f"  {sig.get('field','?')}: Qwen d={sig['qwen_d']:+.2f}, Llama d={sig['llama_d']:+.2f}")

# Layer peaks
c("Llama peak layer", 23, F.get("layer_peak_llama", {}).get("layer", -1), 0.0)
c("Llama peak layer d", 2.81, F.get("layer_peak_llama", {}).get("d", 0), 0.02)

# Vision norms
c("Qwen vision d", 0.56, F.get("vision_norm_qwen", {}).get("d", 0))
c("Llama vision d", 0.67, F.get("vision_norm_llama", {}).get("d", 0))

# Matched complexity
mc = F.get("matched_complexity", {})
c("Matched d", -0.40, mc.get("d", 0))

reg = F.get("regression", {})
c("Regression M1 R2 (culture only)", 0.050, reg["r2_culture"], 0.002)
c("Regression M2 R2 (+compression)", 0.067, reg["r2_plus_comp"], 0.002)
c("Regression M3 R2 (+VLM signals)", 0.270, reg["r2_full"], 0.002)
c("Regression M4 R2 (no culture)", 0.265, reg["r2_no_culture"], 0.002)
c("Regression unique culture R2", 0.005, reg["unique_culture"], 0.001)

med = F.get("mediation", {})
c("Mediation proportion (%)", 8.8, med["proportion_pct"], 0.5)

print(f"\n  Part A checks: {checks}")

print("\n" + "=" * 70)
print("  PART B: EXTENDED STATS (from extended_stats.json)")
print("=" * 70)

# Temperature
T = E["temperature"]
for model in ["qwen", "llama"]:
    for temp in ["T=0.0", "T=0.5", "T=1.0"]:
        d = T[model][temp]
        print(f"  {model} {temp}: EA={d['ea_mean']:.3f} W={d['w_mean']:.3f} "
              f"d={d['d']:+.3f} p={d['p']:.2e} n_EA={d['ea_n']} n_W={d['w_n']}")

# Temperature table verification
c("Tab Q T0 EA", 6.94, T["qwen"]["T=0.0"]["ea_mean"], 0.005)
c("Tab Q T0 W", 7.23, T["qwen"]["T=0.0"]["w_mean"], 0.005)
c("Tab Q T0 d", -0.46, T["qwen"]["T=0.0"]["d"])
pb("Tab Q T0", -27, T["qwen"]["T=0.0"]["p"])

c("Tab Q T5 EA", 6.97, T["qwen"]["T=0.5"]["ea_mean"], 0.005)
c("Tab Q T5 W", 7.25, T["qwen"]["T=0.5"]["w_mean"], 0.005)
c("Tab Q T5 d", -0.44, T["qwen"]["T=0.5"]["d"])
pb("Tab Q T5", -24, T["qwen"]["T=0.5"]["p"])

c("Tab Q T10 EA", 6.97, T["qwen"]["T=1.0"]["ea_mean"], 0.005)
c("Tab Q T10 W", 7.23, T["qwen"]["T=1.0"]["w_mean"], 0.005)
c("Tab Q T10 d", -0.38, T["qwen"]["T=1.0"]["d"])
pb("Tab Q T10", -18, T["qwen"]["T=1.0"]["p"])

c("Tab L T0 EA", 7.56, T["llama"]["T=0.0"]["ea_mean"], 0.005)
c("Tab L T0 W", 7.89, T["llama"]["T=0.0"]["w_mean"], 0.005)
c("Tab L T0 d", -0.36, T["llama"]["T=0.0"]["d"])
pb("Tab L T0", -17, T["llama"]["T=0.0"]["p"])

c("Tab L T5 EA", 7.56, T["llama"]["T=0.5"]["ea_mean"], 0.005)
c("Tab L T5 W", 7.83, T["llama"]["T=0.5"]["w_mean"], 0.005)
c("Tab L T5 d", -0.24, T["llama"]["T=0.5"]["d"])
pb("Tab L T5", -7, T["llama"]["T=0.5"]["p"])

c("Tab L T10 EA", 7.11, T["llama"]["T=1.0"]["ea_mean"], 0.005)
c("Tab L T10 W", 7.55, T["llama"]["T=1.0"]["w_mean"], 0.005)
c("Tab L T10 d", -0.33, T["llama"]["T=1.0"]["d"])
pb("Tab L T10", -13, T["llama"]["T=1.0"]["p"])

# Cross-attention
CA = E["cross_attention"]
c("CA d", 1.30, CA["mean_entropy"]["d"])
pb("CA p", -178, CA["mean_entropy"]["p"])
c("CA EA", 4.99, CA["mean_entropy"]["ea_mean"])
c("CA W", 4.85, CA["mean_entropy"]["w_mean"])
c("CA peak layer", 2, CA["per_layer"]["peak_layer"], 0.0)
c("CA peak d", 1.44, CA["per_layer"]["peak_d"])

# All layers d > 0.5?
all_above = all(d > 0.5 for d in CA["per_layer"]["layer_ds"])
checks += 1
if not all_above:
    errors.append("  FAIL: Not all cross-attn layers have d > 0.5")
print(f"  Cross-attn all layers d>0.5: {all_above}")

# Gradient
G = E["gradient_details"]
c("Grad mean d", 0.26, G["aesthetic_grad_norm_mean"]["d"])
pb("Grad mean p", -9, G["aesthetic_grad_norm_mean"]["p"])
c("Grad mean EA", 12.80, G["aesthetic_grad_norm_mean"]["ea_mean"])
c("Grad mean W", 10.00, G["aesthetic_grad_norm_mean"]["w_mean"])
c("Grad std d", 0.18, G["aesthetic_grad_norm_std"]["d"])
pb("Grad std p", -4, G["aesthetic_grad_norm_std"]["p"])
c("Grad max d", 0.09, G["aesthetic_grad_norm_max"]["d"])
ratio = (G["aesthetic_grad_norm_mean"]["ea_mean"] / G["aesthetic_grad_norm_mean"]["w_mean"] - 1) * 100
c("Grad 28% claim", 28, ratio, 1.5)

# Logit distribution
L = E["logit_distribution"]
c("Qwen KL", 0.0011, L["qwen"]["kl_ea_w"], 0.0002)
c("Llama KL", 0.0007, L["llama"]["kl_ea_w"], 0.0002)

# Encoding robustness table
ER = E["encoding_robustness"]
c("ER Q r0 d", 0.78, ER["qwen"]["ref0"]["d"])
c("ER L r0 d", 0.33, ER["llama"]["ref0"]["d"])
pb("ER Q r0 p", -72, ER["qwen"]["ref0"]["p"])
pb("ER L r0 p", -14, ER["llama"]["ref0"]["p"])
c("ER Q r1 d", -0.22, ER["qwen"]["ref1"]["d"])
c("ER L r1 d", -0.12, ER["llama"]["ref1"]["d"])
pb("ER Q r1 p", -6, ER["qwen"]["ref1"]["p"])
c("ER Q r2 d", -1.08, ER["qwen"]["ref2"]["d"])
c("ER L r2 d", -0.12, ER["llama"]["ref2"]["d"])
pb("ER Q r2 p", -130, ER["qwen"]["ref2"]["p"])
c("ER Q r3 d", -0.51, ER["qwen"]["ref3"]["d"])
c("ER L r3 d", -0.37, ER["llama"]["ref3"]["d"])
pb("ER Q r3 p", -33, ER["qwen"]["ref3"]["p"])
pb("ER L r3 p", -18, ER["llama"]["ref3"]["p"])
c("ER L r1 p exact", 5.4e-3, ER["llama"]["ref1"]["p"], 0.1e-3)
c("ER L r2 p exact", 4.6e-3, ER["llama"]["ref2"]["p"], 0.1e-3)

# Abstract claim: "all p < 10^-7"
min_p_exp = -300
for model in ["qwen", "llama"]:
    for temp in ["T=0.0", "T=0.5", "T=1.0"]:
        import math
        log_p = math.log10(T[model][temp]["p"]) if T[model][temp]["p"] > 0 else -300
        if log_p > min_p_exp:
            min_p_exp = log_p
print(f"  Largest temperature p-value: 10^{min_p_exp:.1f}")
checks += 1
if min_p_exp > -7:
    errors.append(f"  FAIL: Abstract claims 'all p<10^-7' but largest p = 10^{min_p_exp:.1f}")

# 18% reduction claim
d0 = abs(T["qwen"]["T=0.0"]["d"])
d1 = abs(T["qwen"]["T=1.0"]["d"])
pct = (d0 - d1) / d0 * 100
c("18% reduction", 18, pct, 1.0)

# KL < 0.002 claim (abstract)
checks += 1
if L["qwen"]["kl_ea_w"] >= 0.002 or L["llama"]["kl_ea_w"] >= 0.002:
    errors.append(f"  FAIL: KL < 0.002 claim")

print(f"\n  Part B checks: {checks - 15}")  # rough

print("\n" + "=" * 70)
print(f"  GRAND TOTAL: {checks} checks, {len(errors)} errors")
print("=" * 70)
if errors:
    for e in errors:
        print(e)
else:
    print("  *** ALL CHECKS PASSED ***")
