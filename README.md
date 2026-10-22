# Visual reliance in medical VQA — reproducibility repository

This repository accompanies the empirical illustration in:

> *From Radiomics to Multimodal Large Language Models: A Critical Review of Fusion, Grounding, Reasoning, and Clinical Evidence* (manuscript IBMED-D-26-00017).

Repository: <https://github.com/Nesma876/From-Radiomics-to-Multimodal-Large-Language-Models>

The experiment separates visual-input sensitivity from task-specific visual utility on closed yes/no questions from VQA-RAD and SLAKE. It compares question-only, early-fusion and co-attention supervised heads with zero-shot MedGemma-4B. These benchmark results do **not** establish spatial grounding, clinical reasoning, or patient benefit.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Kaggle/local pipeline for environment checks, data preparation, model execution and exploratory analysis. |
| `notebooks/` | Kaggle orchestration notebook. |
| `docs/` | Original analysis plan and Kaggle execution guide. |
| `reproducibility/` | Audited publication artifacts: derived manifests, predictions, model/run metadata, resource measurements, reported statistics, figure sources and independent verification code. |
| `data/` | Provenance and download instructions only. No medical images or original dataset archives are included. |

The publication-facing values are those in `reproducibility/`. The initial `src/04_analysis.py` is retained to document the original Kaggle pipeline, but its older question-level resampling/McNemar implementation is not the authoritative implementation of the final image-cluster inference. See [`docs/STATISTICAL_PROVENANCE.md`](docs/STATISTICAL_PROVENANCE.md).

## Verify the published point estimates

Create an environment and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install pandas numpy scipy
python reproducibility/code/verify_results.py \
  --outputs reproducibility \
  --out reproducibility/statistics/independently_verified_metrics.csv
```

The verifier reconstructs the six primary visual-utility point estimates and prediction-change rates exactly from the released prediction tables. Independently seeded image-cluster bootstrap and permutation results agree with the archived results within the documented Monte-Carlo tolerance; bit-for-bit reproduction of the archived resampling stream is not claimed.

## Re-run the model pipeline

The full pipeline requires users to obtain the datasets from their authorized upstream distributions and to accept the MedGemma terms. Start with [`docs/GUIDE_KAGGLE.md`](docs/GUIDE_KAGGLE.md). Paths in the original notebook may need to be adjusted from `expe/` to this repository layout.

Important distinctions:

- conventional heads were trained with seeds 0--4 using a frozen BiomedCLIP encoder;
- MedGemma-4B was evaluated zero-shot with deterministic decoding;
- model families differ in pretraining, scale, objective and adaptation, so their differences are descriptive rather than causal evidence about fusion architecture;
- no patient identifier was available, so swapped images must not be described as belonging to a different patient.

## Data and redistribution

- VQA-RAD metadata are available from [OSF](https://doi.org/10.17605/OSF.IO/89KPS); images originate from MedPix.
- SLAKE is available from the authors and from `BoKelvin/SLAKE` on Hugging Face. Its source images include material with CC BY-SA 4.0 and CC BY-NC-SA 4.0 conditions.
- This repository redistributes no source medical image and no original dataset archive.
- The upstream RadGrounder split files are excluded until their repository licence is verified. Only derived identifiers/manifests and provenance records needed to audit the reported analysis are included.

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for the complete boundary.

## Software licence

Original code in this repository is released under the MIT License. This does not alter any third-party dataset, model, checkpoint or software licence. Derived documentation and tables may be cited and reused under CC BY 4.0 where the authors hold the relevant rights.

## Citation

Until the article receives its final DOI, cite the manuscript using [`CITATION.cff`](CITATION.cff) and include the repository commit hash used for analysis.

