# FINAL EXPERIMENT VALIDATION — status: **COMPLETE**

## Reused results and justification

```
{
 "vqarad_classical": {
  "decision": "RETRAIN",
  "runs": "q_only, early, coattention x 5 seeds",
  "reason": "mandatory: legacy heads trained on the original (contaminated) split"
 },
 "slake_classical": {
  "decision": "RETRAIN (case B)",
  "reason": "legacy SLAKE predictions cannot be verified (no checkpoints, configs or feature hashes); a clean homogeneous package is produced"
 },
 "medgemma": {
  "slake": {
   "decision": "REUSE_AFTER_PHASE5_CHECKS",
   "candidates": [
    {
     "run": "/kaggle/input/datasets/zineb1993/ibmed-v3-results",
     "file": "/kaggle/input/datasets/zineb1993/ibmed-v3-results/predictions/mllm_medgemma.csv",
     "dataset": "slake",
     "conditions": [
      "correct",
      "gray",
      "no_visual",
      "swapped"
     ]
    }
   ]
  },
  "vqarad": {
   "decision": "REUSE_AFTER_PHASE5_CHECKS",
   "candidates": [
    {
     "run": "/kaggle/input/datasets/zineb1993/ibmed-v3-results",
     "file": "/kaggle/input/datasets/zineb1993/ibmed-v3-results/predictions/mllm_medgemma.csv",
     "dataset": "vqarad",
     "conditions": [
      "correct",
      "gray",
      "no_visual",
      "swapped"
     ]
    }
   ]
  }
 },
 "gpu_benchmarks": {
  "decision": "RE-PROFILE ALL (q_only, early, coattention, medgemma)",
  "reason": "no legacy measurement follows the 5 warm-up + 3 x 50 = 150 protocol"
 },
 "medgemma_missing_conditions": {
  "slake": [],
  "vqarad": []
 }
}
```

## Newly executed

- Classical trainings: 30 runs on ['vqarad', 'slake'] (device cuda, fake encoder: False)
- New MedGemma inferences: 0

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| slake: no train/test image overlap | PASS | 0 shared images |
| vqarad: no train/test image overlap | PASS | 0 shared images |
| sample_id unique | PASS |  |
| slake: classical results from the current split only | PASS | rows with another split_id |
| slake: classical labels match manifest | PASS |  |
| slake/q_only: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| slake/q_only: all conditions | PASS |  |
| slake/q_only: no duplicated predictions | PASS |  |
| slake/early: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| slake/early: all conditions | PASS |  |
| slake/early: no duplicated predictions | PASS |  |
| slake/coattention: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| slake/coattention: all conditions | PASS |  |
| slake/coattention: no duplicated predictions | PASS |  |
| slake: checkpoint early/seed3_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint early/seed2_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint early/seed4_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint early/seed1_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint early/seed0_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint q_only/seed3_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint q_only/seed2_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint q_only/seed4_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint q_only/seed1_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint q_only/seed0_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint coattention/seed3_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint coattention/seed2_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint coattention/seed4_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint coattention/seed1_config trained on current split | PASS | 8e57938223ba7afa |
| slake: checkpoint coattention/seed0_config trained on current split | PASS | 8e57938223ba7afa |
| vqarad: classical results from the current split only | PASS | rows with another split_id |
| vqarad: classical labels match manifest | PASS |  |
| vqarad/q_only: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| vqarad/q_only: all conditions | PASS |  |
| vqarad/q_only: no duplicated predictions | PASS |  |
| vqarad/early: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| vqarad/early: all conditions | PASS |  |
| vqarad/early: no duplicated predictions | PASS |  |
| vqarad/coattention: 5 seeds | PASS | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| vqarad/coattention: all conditions | PASS |  |
| vqarad/coattention: no duplicated predictions | PASS |  |
| vqarad: checkpoint early/seed3_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint early/seed2_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint early/seed4_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint early/seed1_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint early/seed0_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint q_only/seed3_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint q_only/seed2_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint q_only/seed4_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint q_only/seed1_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint q_only/seed0_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint coattention/seed3_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint coattention/seed2_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint coattention/seed4_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint coattention/seed1_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| vqarad: checkpoint coattention/seed0_config trained on current split | PASS | 09ab5d3f2b1dc16b |
| classical features from the real BiomedCLIP encoder | PASS | FAKE encoder (synthetic test) |
| MedGemma predictions complete | PASS | 0 missing |
| MedGemma: no duplicated (sample, condition) | PASS |  |
| MedGemma labels match manifest | PASS |  |
| MedGemma metrics vs announced values | INFO | {"slake": {"accuracy_correct": "reproduced", "change_rate_no_visual": "reproduced", "visual_utility_pp": "reproduced", "invalid_no_visual": "reproduced"}, "vqarad": {"accuracy_correct": "reproduced",  |
| confidence intervals valid (low <= estimate <= high, within +/-100 pp) | PASS |  |
| profiling coattention: exactly 150 timed measurements, all > 0 | PASS | 150 measurements |
| profiling early: exactly 150 timed measurements, all > 0 | PASS | 150 measurements |
| profiling medgemma: exactly 150 timed measurements, all > 0 | PASS | 150 measurements |
| profiling q_only: exactly 150 timed measurements, all > 0 | PASS | 150 measurements |
| profiling covers the four models | PASS | missing [] |
| MedGemma revision recorded (or explicitly unknown) | PASS | 290cda5eeccbee130f987c4ad74a59ae6f196408 |
| LaTeX tables / figure consistent with CSV | PASS | [] |
| information for Supplement S2 present | PASS | missing [] |

## Final statistics (primary contrast)

| dataset   | model       |   utility_pp |   ci_low_pp |   ci_high_pp |   change_rate_pct |   p_permutation |
|:----------|:------------|-------------:|------------:|-------------:|------------------:|----------------:|
| slake     | q_only      |         0    |        0    |         0    |            nan    |          nan    |
| slake     | early       |        -2.03 |       -5.71 |         1.47 |             21.63 |            0.27 |
| slake     | coattention |        23.49 |       19.32 |        27.62 |             31.83 |            0    |
| slake     | medgemma    |        44.51 |       37.85 |        51.27 |             59.15 |            0    |
| vqarad    | q_only      |         0    |        0    |         0    |            nan    |          nan    |
| vqarad    | early       |         3.16 |       -4.6  |        10.92 |             31.33 |            0.44 |
| vqarad    | coattention |         2.91 |       -4    |         9.58 |             37.4  |            0.42 |
| vqarad    | medgemma    |        17.86 |       10.47 |        25.26 |             46.43 |            0    |

## GPU resources

| model       | checkpoint                                                               |   total_parameters_nominal |   trainable_parameters |   stored_elements | quantization           | precision                         |   weights_resident_mb |   peak_allocated_mb |   peak_reserved_mb |   training_seconds_median_per_run |   training_seconds_total |   embedding_prep_seconds | latency_scope                                                                            |   n_warmup |   n_timed |   latency_mean_ms |   latency_median_ms |   latency_p95_ms | gpu      |
|:------------|:-------------------------------------------------------------------------|---------------------------:|-----------------------:|------------------:|:-----------------------|:----------------------------------|----------------------:|--------------------:|-------------------:|----------------------------------:|-------------------------:|-------------------------:|:-----------------------------------------------------------------------------------------|-----------:|----------:|------------------:|--------------------:|-----------------:|:---------|
| q_only      | slake seed 0 (representative; heads differ only in weights across seeds) |                  196034306 |                 131585 |         196034306 | none                   | float32 (encoder fp32, head fp32) |                750.28 |              768.34 |                840 |                              9.41 |                    88.85 |                   125.23 | text tokenisation + BiomedCLIP text encoding + head (no image input)                     |          5 |       150 |             24.72 |               24.41 |            27.76 | Tesla T4 |
| early       | slake seed 0 (representative; heads differ only in weights across seeds) |                  196165378 |                 262657 |         196165378 | none                   | float32 (encoder fp32, head fp32) |                750.78 |              768.84 |                840 |                              9.4  |                    85.61 |                   125.23 | image load + preprocessing + BiomedCLIP image/text encoding + head                       |          5 |       150 |             67.93 |               60.93 |            83.02 | Tesla T4 |
| coattention | slake seed 0 (representative; heads differ only in weights across seeds) |                  196955394 |                1052673 |         196955394 | none                   | float32 (encoder fp32, head fp32) |                753.79 |              771.86 |                844 |                             18.6  |                   231.23 |                   125.23 | image load + preprocessing + BiomedCLIP image/text encoding + head                       |          5 |       150 |             69.03 |               61.3  |            85.45 | Tesla T4 |
| medgemma    | google/medgemma-4b-it @ 290cda5eeccbee130f987c4ad74a59ae6f196408         |                 4300079472 |                      0 |        2490222960 | bitsandbytes 4-bit NF4 | float32 compute                   |               4549.16 |             4807.47 |               4948 |                              0    |                     0    |                     0    | image load + chat template + processor + greedy generation (max 8 new tokens) + decoding |          5 |       150 |           3261.56 |             3258.97 |          3360.88 | Tesla T4 |

## Unresolved problems

- none

## Files needed for publication

- `configs/protocol.json`, `data_splits/vqa_rad_fixed/*`, all scripts in `src/`
- `outputs/manifests/*` (manifests, provenance, exclusions, swap mapping, interventions)
- `outputs/predictions/classical_predictions.csv`, `outputs/predictions/medgemma_predictions.parquet` (+ .csv)
- `outputs/statistics/*.csv`, `outputs/compute/*`, `outputs/manuscript/*`
- `outputs/logs/run_manifest.jsonl` (execution records with output hashes)