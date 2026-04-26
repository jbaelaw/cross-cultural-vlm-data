# Cross-Cultural Score Disparities in Vision-Language Models

Companion data and code for the manuscript:

> **Cross-Cultural Score Disparities in Vision-Language Models: Internal Probing Reveals Processing Asymmetry**

## Quick start

```bash
pip install -r requirements.txt
python code/verify_complete.py
```

`verify_complete.py` cross-checks 71 numerical claims from the manuscript against the JSON files in `data/`. Expected output: `*** ALL CHECKS PASSED ***`.

## Repository contents

- `code/` — analysis scripts
  - `vlm_inference_total.py` — 6-pass VLM inference pipeline (generation, attention, hidden states, encoding cost, gradient attribution, vision encoder, cross-attention)
  - `final_analysis.py` — core statistical analysis (effect sizes, regression, mediation, layer profiles)
  - `extended_analysis.py` — temperature, logit distributions, cross-attention, encoding robustness, gradient details
  - `generate_figures.py` — publication figure generation
  - `verify_complete.py` — numerical verification of manuscript values against derived statistics (no GPU required)
- `data/` — aggregated derived data (JSON)
  - `final_stats.json` — core statistics (effect sizes, p-values, sample sizes, regression, mediation)
  - `extended_stats.json` — temperature, logit, cross-attention, encoding robustness, gradient details
  - `experiment_metadata_all.json` — image-to-culture mapping for all 2,328 records analyzed
  - `compression_curated.json` — per-image compression and spectral metrics
- `figures/` — figure PDFs as published
- `prompts/` — verbatim inference prompts and reference texts

## Reproducing manuscript values

`verify_complete.py` reads the JSON files in `data/` and reproduces the 71 numerical claims (effect sizes, p-values, sample sizes, regression coefficients, etc.) reported in the main text and supplementary tables.

`final_analysis.py`, `extended_analysis.py`, and `generate_figures.py` regenerate the JSON files and figure PDFs from the raw per-image result shards. Those shards are not redistributed in this lightweight GitHub package; they are deposited in the companion Zenodo archive. To run those scripts end-to-end, download the Zenodo archive and place its `derived_results/runpod_final/` directory at `results/runpod_final/` under this repository.

## Dataset notes

- 2,328 records were analyzed: East Asian (n = 1,200; Korean 300, Chinese 500, Japanese 400) and Western (n = 1,128).
- These 2,328 records correspond to 2,313 unique images. Fifteen images appear under two cultural folders during dataset assembly (e.g., works held in cross-regional collections at the Metropolitan Museum of Art) and were therefore included as separate records under each label, exactly as described in the manuscript.
- Source images are not redistributed in this package. They are available under the Open Access policies of the Metropolitan Museum of Art, the National Museum of Korea, and the Cleveland Museum of Art, in accordance with CC0 licensing or Korea's public-API regulations.
- Raw VLM hidden-state arrays and raw vision-encoder embedding arrays are not included; they are available from the corresponding author on request.

## License

- Code: MIT (see `LICENSE`).
- Data files in `data/`, `prompts/`, and `figures/`: CC0 1.0 (see `data/LICENSE`), to the extent permitted by the underlying source-data licenses.

## Citation

See `CITATION.cff`.
