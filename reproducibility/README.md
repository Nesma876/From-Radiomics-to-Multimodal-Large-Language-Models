# S2 reproducibility package

This package accompanies the controlled medical-VQA illustration in Part S2 of the single Supplementary Material document. It contains the supplied audited manifests, predictions, statistics, resource measurements, training metadata, and an independent verification script. It contains no source images or patient-level clinical data.

## Reproduce the numerical audit

From the package directory, with Python, NumPy, and pandas installed:

```text
python code/verify_results.py --outputs . --out statistics/independently_verified_metrics.csv
```

The script requires visual-utility point estimates and prediction-change rates to match the archived outputs exactly. Confidence limits and permutation p-values are checked within explicit Monte-Carlo tolerances because the exact source code and random-number stream used to create the archived inferential files were not present in the supplied archive. Use `--strict-archived-inference` to request a strict archived-inference check; this limitation means bit-for-bit reproduction of those resampled quantities is not claimed.

## Directory contents

- `manifests/`: eligibility, exclusions, split provenance/validation, conditions, interventions, and deterministic swap mapping.
- `predictions/`: raw classical and MedGemma prediction tables in CSV form.
- `statistics/`: archived numerical outputs and independent verification outputs.
- `compute/`: raw and summarized resource measurements and environment/parameter manifests.
- `training_metadata/`: five-seed configurations and histories; binary checkpoints are preserved in the working archive but omitted here to keep the submission package compact.
- `run_metadata/`: execution manifest and audit reports.
- `figure/`: editable SVG and publication PDF/PNG for the visual-reliance figure.
- `code/verify_results.py`: independent reconstruction of the primary correct-image versus no-visual-information contrast.
- `code/generate_visual_reliance_figure.py`: dependency-free reconstruction of an editable SVG from the archived primary-effect table.
- `SHA256SUMS.csv`: integrity inventory for every packaged file except itself.

## Scope and limitations

The experiment is restricted to closed yes/no questions from VQA-RAD and SLAKE. Classical and generative systems have unmatched pretraining, scale, objectives, and adaptation regimes. “No visual information” is implemented as a mean training-image embedding for classical heads and a text-only prompt for MedGemma, so utilities are interpreted within model. Image-clustered inference does not establish patient-level independence. These benchmark results do not establish localization, faithful grounding, clinical reasoning, or patient benefit.

## Dataset access

VQA-RAD question-answer metadata are available from OSF (DOI 10.17605/OSF.IO/89KPS; CC0 metadata; images originate from MedPix). SLAKE is available from the authors' distribution and the first-author Hugging Face distribution (`BoKelvin/SLAKE`, declared CC BY 4.0), while source-image terms include CC BY-SA 4.0 for Medical Segmentation Decathlon and CC BY-NC-SA 4.0 for CHAOS; NIH ChestX-ray14 attribution requirements also apply. No source images are redistributed. See `LICENSE_AND_REDISTRIBUTION.md`.

