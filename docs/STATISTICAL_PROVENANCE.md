# Statistical provenance

The model-execution archive and the final publication audit were completed at different stages.

## Authoritative publication artifacts

Files under `reproducibility/statistics/`, together with `reproducibility/code/verify_results.py`, correspond to the manuscript and Supplementary Material, Part S2. The primary correct-image versus no-visual-information contrast uses:

- 10,000 paired bootstrap resamples clustered by image;
- a 10,000-resample image-cluster sign-flip permutation test;
- Holm correction across the six multimodal model–dataset comparisons;
- invalid generated answers counted as incorrect in the primary analysis;
- valid-output-only and exact-duplicate exclusion sensitivity analyses.

Question-level exact McNemar tests are secondary diagnostics because multiple questions can share an image.

## Legacy pipeline script

`src/04_analysis.py` is retained from the Kaggle execution package for provenance. It uses an earlier question-level bootstrap and McNemar/Holm workflow and must not be used to overwrite the final clustered inferential results reported in the paper. It remains useful for reconstructing exploratory tables and figures from a fresh model run.

The archived resampling results cannot be reproduced bit-for-bit because the exact original analysis source and random-number stream were absent from the supplied output archive. Independent recomputation agrees within the documented Monte-Carlo tolerances and does not change any conclusion.

