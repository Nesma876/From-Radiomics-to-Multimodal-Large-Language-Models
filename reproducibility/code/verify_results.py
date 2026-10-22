"""Independent reconstruction of the six primary empirical contrasts.

Uses only the archived per-question predictions and dataset manifests. Classical
estimates average per-question correctness/change over the five observed seeds;
seeds are not treated as independent statistical units. Resampling and sign
flips operate on image clusters. Invalid MedGemma outputs remain incorrect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_RESAMPLES = 10_000
MODELS = ("early", "coattention", "medgemma")
DATASETS = ("slake", "vqarad")


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def clustered_bootstrap(values: pd.Series, clusters: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"value": values.to_numpy(float), "cluster": clusters.to_numpy()})
    groups = {key: group.value.to_numpy() for key, group in frame.groupby("cluster", sort=True)}
    keys = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(SEED)
    estimates = np.empty(N_RESAMPLES)
    for index in range(N_RESAMPLES):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        estimates[index] = np.concatenate([groups[key] for key in sampled]).mean()
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def cluster_sign_flip(values: pd.Series, clusters: pd.Series) -> float:
    frame = pd.DataFrame({"value": values.to_numpy(float), "cluster": clusters.to_numpy()})
    sums = frame.groupby("cluster", sort=True).value.sum().to_numpy()
    observed = abs(sums.sum() / len(frame))
    rng = np.random.default_rng(SEED)
    extreme = 0
    for _ in range(N_RESAMPLES):
        statistic = abs((sums * rng.choice((-1, 1), size=len(sums))).sum() / len(frame))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (N_RESAMPLES + 1)


def load_question_level(root: Path, dataset: str, model: str) -> pd.DataFrame:
    manifest = pd.read_csv(root / "manifests" / f"{dataset}_manifest.csv")
    image_map = manifest.set_index("sample_id")["image_pixel_hash"]
    if model == "medgemma":
        raw = pd.read_csv(root / "predictions" / "medgemma_predictions.csv")
        raw = raw[(raw.dataset == dataset) & (raw.model == model)].copy()
        raw["correctness"] = ((raw.pred == raw.label) & raw.pred.isin([0, 1])).astype(float)
        table = raw.pivot(index="sample_id", columns="condition", values=["correctness", "pred"])
        result = pd.DataFrame(index=table.index)
        result["correct"] = table["correctness", "correct"]
        result["no_visual"] = table["correctness", "no_visual"]
        result["changed"] = (table["pred", "correct"] != table["pred", "no_visual"]).astype(float)
    else:
        raw = pd.read_csv(root / "predictions" / "classical_predictions.csv")
        raw = raw[(raw.dataset == dataset) & (raw.model == model)].copy()
        raw["correctness"] = (raw.pred == raw.label).astype(float)
        correct = raw[raw.condition == "correct"].set_index(["sample_id", "seed"])
        absent = raw[raw.condition == "no_visual"].set_index(["sample_id", "seed"])
        joined = correct[["correctness", "pred"]].join(
            absent[["correctness", "pred"]], lsuffix="_correct", rsuffix="_no_visual", validate="one_to_one"
        )
        per_question = joined.groupby(level="sample_id")
        result = pd.DataFrame(index=sorted(joined.index.get_level_values("sample_id").unique()))
        result["correct"] = per_question.correctness_correct.mean()
        result["no_visual"] = per_question.correctness_no_visual.mean()
        result["changed"] = per_question.apply(
            lambda group: (group.pred_correct != group.pred_no_visual).mean(), include_groups=False
        )
    result["image_hash"] = image_map.reindex(result.index)
    if result.image_hash.isna().any():
        raise ValueError(f"Missing image hashes for {dataset}/{model}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--strict-archived-inference",
        action="store_true",
        help="Fail unless Monte-Carlo confidence limits and permutation p-values exactly reproduce the archived run.",
    )
    args = parser.parse_args()
    root = args.outputs.resolve()
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            question = load_question_level(root, dataset, model)
            difference = question.correct - question.no_visual
            low, high = clustered_bootstrap(difference, question.image_hash)
            rows.append({
                "dataset": dataset,
                "model": model,
                "n_questions": len(question),
                "n_images": question.image_hash.nunique(),
                "correct_accuracy": question.correct.mean(),
                "no_visual_accuracy": question.no_visual.mean(),
                "visual_utility_pp": 100 * difference.mean(),
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "prediction_change_pct": 100 * question.changed.mean(),
                "p_permutation": cluster_sign_flip(difference, question.image_hash),
            })
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame.p_permutation.tolist())

    archived = pd.read_csv(root / "statistics" / "paired_effects.csv")
    archived = archived[(archived.contrast == "correct - no_visual") & archived.model.isin(MODELS)]
    comparison = frame.merge(
        archived[["dataset", "model", "utility_pp", "ci_low_pp", "ci_high_pp", "change_rate_pct", "p_permutation"]],
        on=["dataset", "model"], suffixes=("_recomputed", "_archived"), validate="one_to_one"
    )
    comparisons = {
        "visual_utility_pp": ("visual_utility_pp", "utility_pp", 1e-10),
        "ci_low_pp": ("ci_low_pp_recomputed", "ci_low_pp_archived", 0.25),
        "ci_high_pp": ("ci_high_pp_recomputed", "ci_high_pp_archived", 0.25),
        "prediction_change_pct": ("prediction_change_pct", "change_rate_pct", 1e-10),
        "p_permutation": ("p_permutation_recomputed", "p_permutation_archived", 0.01),
    }
    checks = {}
    for name, (recomputed_name, archived_name, tolerance) in comparisons.items():
        left = comparison[recomputed_name].to_numpy(float)
        right = comparison[archived_name].to_numpy(float)
        checks[name] = bool(np.all(np.abs(left - right) <= tolerance))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    status = {
        "source": root.name,
        "bootstrap_seed": SEED,
        "resamples": N_RESAMPLES,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "interpretation": (
            "Point estimates and change rates are required to match exactly. "
            "Confidence limits and permutation p-values are accepted only within explicit "
            "Monte-Carlo tolerances because the exact archived analysis source was not supplied."
        ),
    }
    args.out.with_suffix(".json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(status, indent=2))
    if not status["all_checks_pass"] or (
        args.strict_archived_inference
        and not all(checks[name] for name in ("ci_low_pp", "ci_high_pp", "p_permutation"))
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
