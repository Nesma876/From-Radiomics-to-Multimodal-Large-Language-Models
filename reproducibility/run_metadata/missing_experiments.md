# Missing experiments (phase 1 audit, no GPU used)

Legacy runs found: 1

- `/kaggle/input/datasets/zineb1993/ibmed-v3-results`: VQA-RAD split heuristic = image-disjoint (candidate)

## To run

1. **VQA-RAD classical**: retrain q_only, early fusion, co-attention x 5 seeds on the image-disjoint split (15 runs).
2. **SLAKE classical**: retrain the three models x 5 seeds (15 runs), legacy predictions not verifiable.
3. **Classical evaluation**: 4 visual conditions for every checkpoint.
4. **MedGemma**: reuse after phase-5 checks; missing conditions by dataset: {'slake': [], 'vqarad': []}; datasets without reusable predictions: none. Only missing (sample_id, condition) pairs are generated.
5. **GPU profiling**: all four models, new protocol (5 warm-up, 3 x 50 = 150 timed).
6. **Model revisions and parameter audit** (MedGemma reloaded, no training).
