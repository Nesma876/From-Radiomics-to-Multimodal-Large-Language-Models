# Pre-specified analysis plan — version 3

**Study:** empirical illustration of the evaluation protocol proposed in the review IBMED-D-26-00017.
**Version 3 fixed on:** 20 October 2026.

| Version | Date | Status |
| --- | --- | --- |
| 1 | 19 Oct 2026 | Initial plan |
| 2 | 19 Oct 2026 | Refocused on availability / sensitivity / utility, after co-author review and before any real-data run |
| 3 | 20 Oct 2026 | Co-author methodological review after a **pilot run** (see the deviation log at the end) |

Changes in version 3 were motivated by design concerns raised in review: selection bias, unequal neutralization procedures, and split contamination. They were not motivated by the direction of pilot results. They are nevertheless reported as deviations, and the pilot results on the original VQA-RAD split are reported as a sensitivity analysis.

## Research question

> **Does access to a medical image translate into measurable visual reliance, and at what computational cost?**

The study does **not** aim to rank models. Classical fusion classifiers are trained with supervision on the training split. The MLLM is used zero-shot. The two are evaluated under **identical conditions** (questions, split, metric, perturbations, hardware) but under **different learning regimes**. Between-model accuracy differences are therefore secondary and descriptive.

## Three properties, measured separately

| Property | Definition | How it is measured |
| --- | --- | --- |
| **Visual availability** | The model receives an image as input | Architectural property, reported per model (yes/no, and the input interface) |
| **Visual sensitivity** | The prediction changes when the image is altered | Flip rate between `real` and each altered condition. Label-free |
| **Visual utility** | The correct image improves answer accuracy | Δ_visual = acc(`real`) − acc(`none`), with a paired 95 % CI and an exact McNemar test |

These properties can dissociate:

- **Sensitivity without utility.** Answers change with the image, but accuracy does not improve, or even falls.
- **Low sensitivity with no image neglect.** Some questions can be solved from text alone. Low sensitivity is therefore interpreted within question strata (see below).

A change of answer is never, by itself, evidence of medically relevant visual reasoning.

## Data

- **VQA-RAD.**
    - Images and metadata (`question_type`, `image_organ`, `qid`) come from the official OSF release.
    - **Primary analysis:** the **image-disjoint split** of Wu et al. (2025, *Nat Commun* 16:7866), distributed by the `lmb-freiburg/radgrounder` repository (`train/validation/test_fixed_split.json`, bundled in `data_splits/vqa_rad_fixed/`). The original VQA-RAD split shares images between train and test: in the pilot, 227 of 251 yes/no test questions concerned an image also seen in training.
    - Split questions are matched to OSF records by image name and normalized question text. The number of unmatched questions is reported.
    - **Sensitivity analysis only:** the original split (`--vqarad_split official`).
    - The HuggingFace version is not used for the primary analysis.
- **SLAKE 1.0, English subset.** Official train/validation/test files. Metadata used: `content_type`, `modality`, `location`.
- **Inclusion.** Questions whose reference answer, lowercased and with punctuation stripped, is exactly `yes` or `no`. No other filtering.
- **Checks reported, not corrected:**
    - test images also present in train/validation (pixel-identical);
    - duplicate (image, question) pairs within the test split;
    - paraphrase pairs in the VQA-RAD test (`test_freeform` / `test_para`);
    - class balance per split.
- **Licences.** Verify and cite the licence of each release before submission. Record them in `data_summary.csv` notes.
- **Validation split.** SLAKE: official. VQA-RAD: the validation split of the image-disjoint split. With the original split only, 10 % of training images are held out (grouped by image, seed 2026).
- **Released identifiers.** Every included question identifier, its split and its swap image are released in `questions.csv`. Exclusions (non yes/no answers) are counted per split in `data_summary.csv`.

## Models

| ID | Description | Learning regime |
| --- | --- | --- |
| `majority` | Majority training label | — |
| `q_only` | BiomedCLIP question embedding (frozen text encoder) → MLP. No image | Supervised (train split) |
| `early` | BiomedCLIP image embedding ⊕ question embedding → MLP | Supervised |
| `intermediate` | Bidirectional co-attention between BiomedCLIP image patch tokens and question tokens → MLP | Supervised |
| `mllm_medgemma` | `google/medgemma-4b-it`, zero-shot (primary MLLM) | Zero-shot |
| `mllm_llava_med` | `microsoft/llava-med-v1.5-mistral-7b`, zero-shot (optional) | Zero-shot |

- **BiomedCLIP** stays frozen for all classical models.
- **Classical models.** 5 seeds (0–4). Fixed hyperparameters (`common.py`). Early stopping on validation accuracy only. No test-set tuning.
- **MLLM prompt (fixed):** `{question}\nAnswer with yes or no only.`. Greedy decoding, at most 8 new tokens.
- **MLLM parsing.** The first word of the lowercased output, if `yes` or `no`. Otherwise `unparsed`, scored as wrong; its rate is reported per condition.

## Test conditions

| Condition | Meaning | Classical models | MLLM |
| --- | --- | --- | --- |
| `real` | Correct image | Image of the question | Image of the question |
| `none` | **Visual information neutralized** (model-specific procedure) | **Visual-information ablation:** training-set mean of the image embedding and patch tokens | **Text-only prompt** (image input removed). LLaVA-Med: not run, since the model requires an image |
| `gray` | Uninformative image, **potentially out of distribution** | Uniform gray image (RGB 128) through the encoder | Same gray image |
| `swap` | **Misleading, in-distribution** image from another patient | Same swapped image | Same swapped image |

**Swap rule (reproducible).**

- The swap image comes from the **test split of the same dataset**, with the **same modality and anatomical region** when metadata allows: SLAKE `modality` + `location`; VQA-RAD `image_organ`.
- It must be a **different image**.
- It falls back to the dataset-wide test pool if the stratum holds fewer than 2 distinct images. The fallback rate is reported.
- The pairing is drawn once with seed 2026 and shared by all models.

**Neutralization is not the same operation across model families.**

- For MedGemma, the image input is removed.
- For classical fusion, a mean training-image representation is **injected**. This is an approximation of neutralization, not a removal of the visual branch.
- Explicit masking was considered and is not available without changing the trained models:
    - the early-fusion head has no masking path (a fixed-size concatenated input);
    - in the co-attention head, masking every visual key leaves the question-to-image attention undefined.
- Consequently, Δ_visual is computed and interpreted **within each model only**. The two procedures are never compared across families, and tables name the procedure used for each model.

**Why both `none` and `gray`.** `none` targets the removal of image-specific information. `gray` is an uninformative image that may push the encoder out of distribution. Divergence between the two indicates an out-of-distribution effect rather than visual reliance.

## Question strata (interpretation of reliance)

**Primary: annotation-based**, fixed before any model is run and independent of model performance.

1. **Presumed image dependence (SLAKE).** SLAKE annotates each question as vision-based or knowledge-based (`base_type` = `kvqa`, or `content_type` = `KG`). Knowledge-based questions are presumed answerable from medical knowledge. Vision-based questions are presumed to require the image.
2. **Annotated question type.** VQA-RAD `question_type` (e.g., presence, abnormality, organ, modality, plane, position); SLAKE `content_type`.

VQA-RAD has no dependence annotation, so it is analysed by question type only. A clinician annotation of image dependence would strengthen this analysis and is listed as a limitation.

**Exploratory only: text-only baseline solvability.** A question is *text-only-solvable* if `q_only` answers it correctly in ≥ 4 of 5 seeds.

- This is a property of one model, not of the question.
- Defining a stratum from model success introduces selection bias, especially in favour of the text-only baseline.
- It is reported separately, labelled as exploratory, and never used for the main conclusions. `q_only` is excluded from it.

**Expected pattern for genuine visual utility:** Δ_visual is larger on vision-based than on knowledge-based questions.

## Outcomes

**Primary**

- Δ_visual = acc(`real`) − acc(`none`) per dataset × model, with a 95 % paired bootstrap CI (2,000 resamples of questions, seed 2026).
- Exact McNemar test (`real` vs `none`).

**Secondary**

- Accuracy per condition, with 95 % bootstrap CI. Classical models: 5-seed majority vote, plus per-seed mean ± SD.
- Δ_swap = acc(`real`) − acc(`swap`) and Δ_gray = acc(`real`) − acc(`gray`), each with a paired CI.
- Sensitivity: flip rates `real`→`none`, `real`→`gray`, `real`→`swap`. Among flips, the share that corrects an error vs the share that creates one.
- Balanced accuracy and the rate of `yes` answers per condition, because removing the image may shift the answer bias.
- Δ_visual and flip rates per annotated stratum (primary) and per text-only solvability stratum (exploratory).
- Between-model comparison on `real` (`mllm_*` vs `intermediate`, `mllm_*` vs `q_only`): McNemar test plus a paired CI on the difference. Descriptive, given the different learning regimes.

**Multiplicity.** Holm correction within each dataset, in two families:

- the primary family (one `real` vs `none` test per image-using model);
- the secondary family (all other McNemar tests).

## Computational cost (same GPU, standardized)

Reported for every model:

- **Parameters:** total and trainable.
- **Training cost** (classical): wall-clock time and epochs per seed.
- **Inference:** latency at batch size 1, end to end (image preprocessing, encoding, model). Protocol: 5 warm-up questions, then 50 fixed test questions, repeated 3 times. Report the median and interquartile range.
- **Memory:** peak allocated GPU memory.
- **Configuration:** GPU, precision (4-bit nf4 or fp16/bf16), and library versions (from `env.json`).

## Interpretation rules (fixed in advance)

- A model shows **visual utility** on a dataset if Δ_visual > 0 with a Holm-adjusted McNemar p < 0.05 in the primary family.
- A model shows **visual sensitivity** if its `real`→`none` and `real`→`swap` flip rates exceed 5 %. This threshold is descriptive and carries no inferential claim.
- If `q_only` exceeds `majority`, the benchmark is answerable from language priors to that extent. Image-based results are then read against `q_only`.
- No claim is made beyond these two benchmarks and this question format. Benchmark accuracy is not clinical reasoning, clinical utility, or validation.
- Results are reported whatever their direction.

## Limitations to report

- Small test sets.
- Pre-training contamination is possible: check each MLLM model card. A model trained on these train splits is labelled "not strictly zero-shot".
- VQA-RAD: the original split is contaminated (shared images), so the image-disjoint split is used, at the cost of a smaller training set. Paraphrase pairs remain in the data.
- Image dependence is annotated only for SLAKE.
- Different learning regimes (supervised vs zero-shot).
- Neutralization differs by family (mean-representation ablation vs text-only prompt); Δ_visual is interpreted within each model only.
- 4-bit quantization, if used.
- LLaVA-Med v1.5 is not the NeurIPS 2023 model.
- No clinical validation.

## Reproducibility

Scripts, configuration (`common.py`), this plan, `env.json`, the swap table and all prediction files will be released on GitHub and archived on Zenodo with a DOI.


## Deviation log

| Date | Change | Reason | Effect on runs |
| --- | --- | --- | --- |
| 20 Oct 2026 | Primary stratification moved from text-only solvability (model-derived) to dataset annotations; text-only solvability kept as exploratory | Avoid selection bias from defining strata with a model's own success | Analysis only (no rerun) |
| 20 Oct 2026 | `none` renamed per procedure; Δ_visual interpreted within model only | Neutralization procedures are not equivalent across model families | Analysis and reporting only |
| 20 Oct 2026 | VQA-RAD primary analysis on the image-disjoint split (Wu et al., 2025); original split kept as a sensitivity analysis | The original split shares images between train and test | VQA-RAD classical models retrained; MedGemma run on the new VQA-RAD test questions; SLAKE MedGemma predictions reused unchanged (same questions and swap pairs) |
# Archival analysis plan

> This file preserves the prespecified planning record from the execution
> package. Corrective reanalysis subsequently adopted image-clustered bootstrap
> intervals, image-cluster sign-flip permutation tests and a six-comparison Holm
> family. The final procedure is documented in `STATISTICAL_PROVENANCE.md` and
> the publication-facing outputs are under `../reproducibility/`.

