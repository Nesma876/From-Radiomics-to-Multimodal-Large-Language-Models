"""Step 4 — analysis exactly as fixed in ANALYSIS_PLAN.md (version 2).

    python 04_analysis.py

Reads work/predictions/*.csv and work/data/questions.csv. Writes to work/results/:
    properties.csv / .tex        availability, sensitivity, utility per dataset x model (main table)
    accuracy.csv                 accuracy, balanced accuracy, yes-rate, unparsed rate per condition
    deltas.csv                   accuracy differences real - altered, paired 95 % CI
    flips.csv                    flip rates and direction (image helped / image hurt)
    strata_annotated.csv         PRIMARY strata, fixed by dataset annotations before any model run:
                                 presumed image dependence (SLAKE vision- vs knowledge-based) and question type
    strata_exploratory_textonly.csv  EXPLORATORY: text-only baseline solvability (q_only correct in >= 4/5 seeds);
                                 model-derived, hence biased in favour of the text-only baseline
    mcnemar.csv                  pre-specified tests, Holm within dataset x family
    compute_table.csv / .tex     parameters, training cost, latency, memory
    fig_deltas.pdf/.png          accuracy change when the image is removed, grayed or swapped
    fig_sensitivity_utility.pdf/.png   sensitivity (flip rate) vs utility (Delta_visual)
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import BOOT_SEED, CONDITIONS, DATA, N_BOOT, PRED, RES, SEEDS, TEXT_ANSWERABLE_MIN

CLASSICAL = ["q_only", "early", "intermediate"]
IMAGE_MODELS_CLASSICAL = ["early", "intermediate"]
ALTERED = ["none", "gray", "swap"]
# 'none' is implemented differently per model family and is never compared across families:
#   classical fusion -> visual-information ablation (training-set mean image embedding / patch tokens)
#   MLLM             -> text-only prompt (image input removed)
NONE_PROCEDURE = {"early": "visual-information ablation (mean training-image embedding)",
                  "intermediate": "visual-information ablation (mean training-image patch tokens)",
                  "mllm_medgemma": "text-only prompt (image removed)",
                  "mllm_llava_med": "not run (model requires an image)"}
COND_LABEL = {"real": "Correct image", "none": "Visual information neutralized", "gray": "Gray image",
              "swap": "Deterministically swapped image"}
# Validated categorical palette (dataviz skill, light mode); marker shapes give a colour-free cue.
COND_STYLE = {"none": ("#2a78d6", "o"), "gray": ("#eb6834", "s"), "swap": ("#1baf7a", "^")}
DS_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE (English)"}
MODEL_LABEL = {"majority": "Majority", "q_only": "Question only", "early": "Early fusion",
               "intermediate": "Co-attention fusion", "mllm_medgemma": "MedGemma-4B (zero-shot)",
               "mllm_llava_med": "LLaVA-Med 7B (zero-shot)"}
INTERFACE = {"majority": "none", "q_only": "none (by design)", "early": "frozen image embedding",
             "intermediate": "frozen image patch tokens", "mllm_medgemma": "image tokens in the language model",
             "mllm_llava_med": "image tokens in the language model"}
SENS_THRESHOLD = 0.05
# Distinct marker per model (colour-free cue); MLLMs in the two validated accent hues.
MODEL_STYLE = {"early": ("#5c5c57", "o"), "intermediate": ("#5c5c57", "s"),
               "mllm_medgemma": ("#2a78d6", "D"), "mllm_llava_med": ("#eb6834", "^")}


# ======================================================================= loading
def load():
    frames = [pd.read_csv(f) for f in glob.glob(str(PRED / "*.csv"))]
    if not frames:
        raise SystemExit(f"No prediction files in {PRED}")
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = (df["pred"] == df["label"]).astype(int)   # pred = -1 (unparsed) is wrong
    return df


def vote(df):
    """One prediction per question for classical models: majority vote across seeds."""
    v = df.groupby(["dataset", "model", "condition", "qid", "label"])["pred"].mean().reset_index()
    v["pred"] = (v["pred"] >= 0.5).astype(int)
    v["seed"] = -2
    v["correct"] = (v["pred"] == v["label"]).astype(int)
    return v


# ======================================================================= statistics
RNG = np.random.default_rng(BOOT_SEED)


def boot_ci(x):
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired_diff_ci(a, b):
    """95 % CI of mean(a) - mean(b), resampling questions jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return n01, n10, p


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def balanced_acc(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return np.nanmean([tpr, tnr])


# ======================================================================= main
def main():
    raw = load()
    q = pd.read_csv(DATA / "questions.csv", keep_default_na=False).set_index("qid")
    point = pd.concat([raw[~raw["model"].isin(CLASSICAL)], vote(raw[raw["model"].isin(CLASSICAL)])],
                      ignore_index=True)
    mllms = sorted(m for m in point["model"].unique() if m.startswith("mllm_"))
    order = ["majority"] + CLASSICAL + mllms
    datasets = sorted(point["dataset"].unique())

    def get(ds, m, cond):
        s = point[(point["dataset"] == ds) & (point["model"] == m) & (point["condition"] == cond)]
        return s.set_index("qid").sort_index()

    def has(ds, m, cond):
        return not get(ds, m, cond).empty

    # ---------------------------------------------------- EXPLORATORY: text-only baseline solvability
    qo = raw[(raw["model"] == "q_only") & (raw["condition"] == "real")]
    n_ok = qo.groupby("qid")["correct"].sum()
    stratum = pd.Series(np.where(n_ok >= TEXT_ANSWERABLE_MIN, "text-only-solvable", "not text-only-solvable"),
                        index=n_ok.index)
    dep_col = "presumed_image_dependence" if "presumed_image_dependence" in q.columns else None

    acc_rows, delta_rows, flip_rows, strata_rows, qtype_rows, tests = [], [], [], [], [], []
    for ds in datasets:
        for m in order:
            if not has(ds, m, "real"):
                continue
            real = get(ds, m, "real")
            for c in CONDITIONS:
                if not has(ds, m, c):
                    continue
                v = get(ds, m, c)
                lo, hi = boot_ci(v["correct"])
                r = dict(dataset=ds, model=m, condition=c, n=len(v), accuracy=v["correct"].mean(),
                         ci_low=lo, ci_high=hi, balanced_accuracy=balanced_acc(v["label"], v["pred"]),
                         yes_rate=(v["pred"] == 1).mean(), unparsed_rate=(v["pred"] == -1).mean())
                if m in CLASSICAL:
                    ps = raw[(raw["dataset"] == ds) & (raw["model"] == m) & (raw["condition"] == c)
                             ].groupby("seed")["correct"].mean()
                    r["seed_mean"], r["seed_sd"] = ps.mean(), ps.std(ddof=1)
                acc_rows.append(r)
            if m in ("majority", "q_only"):
                continue
            for c in ALTERED:
                if not has(ds, m, c):
                    continue
                a, b = real["correct"].align(get(ds, m, c)["correct"], join="inner")
                pr, pc = real["pred"].align(get(ds, m, c)["pred"], join="inner")
                lo, hi = paired_diff_ci(a, b)
                delta_rows.append(dict(dataset=ds, model=m, comparison=f"real - {c}", delta=a.mean() - b.mean(),
                                       ci_low=lo, ci_high=hi, n=len(a)))
                flip = pr != pc
                flip_rows.append(dict(dataset=ds, model=m, comparison=f"real -> {c}", flip_rate=flip.mean(),
                                      image_helped=int(((a == 1) & (b == 0)).sum()),
                                      image_hurt=int(((a == 0) & (b == 1)).sum()),
                                      flips_changing_correctness=int((a != b).sum()), n_flips=int(flip.sum())))
                if c == "none":
                    def stratum_row(kind, label, ids):
                        lo_s, hi_s = paired_diff_ci(a.loc[ids], b.loc[ids])
                        return dict(dataset=ds, model=m, none_procedure=NONE_PROCEDURE.get(m, ""),
                                    stratification=kind, stratum=label, n=len(ids), acc_real=a.loc[ids].mean(),
                                    acc_none=b.loc[ids].mean(), delta_visual=a.loc[ids].mean() - b.loc[ids].mean(),
                                    ci_low=lo_s, ci_high=hi_s, flip_rate=flip.loc[ids].mean())
                    # PRIMARY: annotation-based, independent of any model's performance
                    if dep_col:
                        dep = q.loc[a.index, dep_col]
                        for t in sorted(set(dep) - {"", "not annotated"}):
                            qtype_rows.append(stratum_row("presumed image dependence (annotation)", t,
                                                          dep.index[dep == t]))
                    qt = q.loc[a.index, "qtype"]
                    for t in sorted(set(qt) - {""}):
                        qtype_rows.append(stratum_row("question type (annotation)", t, qt.index[qt == t]))
                    # EXPLORATORY: model-derived
                    for st in ("text-only-solvable", "not text-only-solvable"):
                        ids = [i for i in a.index if stratum.get(i) == st]
                        if ids:
                            strata_rows.append(stratum_row("text-only baseline solvability (exploratory)", st, ids))

        # ------------------------------------------------ pre-specified tests
        def add_test(family, m1, c1, m2, c2):
            if not (has(ds, m1, c1) and has(ds, m2, c2)):
                return
            a, b = get(ds, m1, c1)["correct"].align(get(ds, m2, c2)["correct"], join="inner")
            n01, n10, p = mcnemar(a.to_numpy(), b.to_numpy())
            lo, hi = paired_diff_ci(a, b)
            tests.append(dict(dataset=ds, family=family, model_1=m1, cond_1=c1, model_2=m2, cond_2=c2,
                              acc_1=a.mean(), acc_2=b.mean(), diff=a.mean() - b.mean(), diff_ci_low=lo,
                              diff_ci_high=hi, only_1_correct=n01, only_2_correct=n10, p=p))
        for m in IMAGE_MODELS_CLASSICAL + mllms:
            add_test("primary", m, "real", m, "none")
        for m in IMAGE_MODELS_CLASSICAL + mllms:
            add_test("secondary", m, "real", m, "gray")
            add_test("secondary", m, "real", m, "swap")
        for mm in mllms:
            add_test("secondary", mm, "real", "intermediate", "real")
            add_test("secondary", mm, "real", "q_only", "real")

    mc = pd.DataFrame(tests)
    if len(mc):
        mc["p_holm"] = mc.groupby(["dataset", "family"])["p"].transform(lambda s: holm(s.to_numpy()))

    acc, deltas, flips = pd.DataFrame(acc_rows), pd.DataFrame(delta_rows), pd.DataFrame(flip_rows)
    for name, d in [("accuracy", acc), ("deltas", deltas), ("flips", flips),
                    ("strata_annotated", pd.DataFrame(qtype_rows)),
                    ("strata_exploratory_textonly", pd.DataFrame(strata_rows)), ("mcnemar", mc)]:
        d.to_csv(RES / f"{name}.csv", index=False)

    # ---------------------------------------------------- properties table (main result)
    props = []
    for ds in datasets:
        for m in order:
            if m == "majority" or not has(ds, m, "real"):
                continue
            r = dict(dataset=ds, model=m, availability=INTERFACE.get(m, "image"),
                     none_procedure=NONE_PROCEDURE.get(m, "n/a"),
                     regime="zero-shot" if m.startswith("mllm_") else "supervised",
                     acc_real=acc.query("dataset == @ds and model == @m and condition == 'real'")["accuracy"].iloc[0])
            for c in ALTERED:
                f = flips[(flips["dataset"] == ds) & (flips["model"] == m) & (flips["comparison"] == f"real -> {c}")]
                r[f"flip_{c}"] = f["flip_rate"].iloc[0] if len(f) else np.nan
            d = deltas.query("dataset == @ds and model == @m and comparison == 'real - none'")
            t = mc.query("dataset == @ds and family == 'primary' and model_1 == @m") if len(mc) else mc
            if len(d):
                r.update(delta_visual=d["delta"].iloc[0], dv_ci_low=d["ci_low"].iloc[0], dv_ci_high=d["ci_high"].iloc[0])
            if len(t):
                r["p_holm"] = t["p_holm"].iloc[0]
            sens = np.nanmax([r.get("flip_none", np.nan), r.get("flip_swap", np.nan)]) if m != "q_only" else 0.0
            r["sensitive"] = "n/a" if m == "q_only" else ("yes" if sens > SENS_THRESHOLD else "no")
            r["utility"] = ("n/a" if m == "q_only" or "p_holm" not in r else
                            "yes" if (r["delta_visual"] > 0 and r["p_holm"] < 0.05) else "not shown")
            props.append(r)
    props = pd.DataFrame(props)
    props.to_csv(RES / "properties.csv", index=False)
    write_latex(props)
    write_compute()
    plot_deltas(deltas, order, datasets)
    plot_sens_util(props, datasets)

    pd.set_option("display.width", 200)
    print(props.round(3).to_string(index=False))
    print()
    sa = pd.DataFrame(qtype_rows)
    if len(sa):
        print(sa[sa["stratification"].str.startswith("presumed")].round(3).to_string(index=False))
    print()
    print(mc.round(4).to_string(index=False))


# ======================================================================= outputs
def fmt_pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def write_latex(props):
    t = props.copy()
    t["Dataset"] = t["dataset"].map(DS_LABEL)
    t["Model"] = t["model"].map(lambda m: MODEL_LABEL.get(m, m))
    t["Regime"] = t["regime"]
    t["Acc. (\\%)"] = t["acc_real"].map(fmt_pct)
    t["Flip none / swap (\\%)"] = [f"{fmt_pct(a)} / {fmt_pct(b)}" for a, b in zip(t["flip_none"], t["flip_swap"])]
    t["$\\Delta_{visual}$ (pp) [95\\% CI]"] = [
        "--" if pd.isna(getattr(r, "delta_visual", np.nan)) else
        f"{100 * r.delta_visual:+.1f} [{100 * r.dv_ci_low:+.1f}, {100 * r.dv_ci_high:+.1f}]"
        for r in t.itertuples()]
    t["$p_{Holm}$"] = t.get("p_holm", pd.Series(np.nan, index=t.index)).map(lambda p: "--" if pd.isna(p) else f"{p:.3f}")
    t["Utility"] = t["utility"]
    cols = ["Dataset", "Model", "Regime", "Acc. (\\%)", "Flip none / swap (\\%)", "$\\Delta_{visual}$ (pp) [95\\% CI]",
            "$p_{Holm}$", "Utility"]
    t[cols].to_latex(RES / "properties.tex", index=False, escape=False,
                     caption="Visual sensitivity and visual utility under identical evaluation conditions. "
                             "Flip: share of answers that change when visual information is neutralized, or when the "
                             "correct image is replaced by another image from the same split. Neutralization is model-specific: "
                             "visual-information ablation with the mean training-image representation for classical "
                             "fusion (an approximation, since these architectures have no explicit masking of the visual "
                             "branch) and a text-only prompt for the MLLM. $\\Delta_{visual}$: accuracy with the correct "
                             "image minus accuracy after neutralization, computed within each model only (paired "
                             "bootstrap CI); the two procedures are not compared across model families. Utility: "
                             "$\\Delta_{visual}>0$ with Holm-adjusted McNemar $p<0.05$. Classical models: 5-seed "
                             "majority vote. Supervised and zero-shot regimes differ; between-model differences "
                             "are descriptive.",
                     label="tab:experiment-properties")


def write_compute():
    comp = [pd.read_csv(f) for f in glob.glob(str(RES / "compute_*.csv")) if "compute_table" not in f]
    if not comp:
        return
    ct = pd.concat(comp, ignore_index=True)
    ct.to_csv(RES / "compute_table.csv", index=False)
    t = pd.DataFrame({
        "Model": ct["model"].map(lambda m: MODEL_LABEL.get(m, m)),
        "Total params (M)": (ct["total_params"] / 1e6).map("{:.1f}".format),
        "Trainable (M)": (ct["trainable_params"] / 1e6).map("{:.2f}".format),
        "Training / seed (s)": ct.get("train_seconds_median_per_seed", pd.Series(np.nan)).map(
            lambda x: "--" if pd.isna(x) or x == 0 else f"{x:.0f}"),
        "Latency, ms [IQR]": [f"{a:.0f} [{b:.0f}--{c:.0f}]" for a, b, c in
                              zip(ct["latency_ms_median"], ct["latency_ms_q1"], ct["latency_ms_q3"])],
        "Peak memory (GB)": (ct["peak_mem_mb"] / 1024).map("{:.1f}".format),
        "Precision": ct["precision"]})
    gpu = ct["gpu"].dropna().iloc[0] if ct["gpu"].notna().any() else "GPU"
    t.to_latex(RES / "compute_table.tex", index=False, escape=True,
               caption=f"Computational cost measured on the same GPU ({gpu}). Latency: batch size 1, end to end, "
                       "5 warm-up questions then 50 fixed questions repeated 3 times; median and interquartile "
                       "range. Training time covers the classification head only (frozen encoder features "
                       "cached once). Parameter counts of 4-bit models are read from the loaded model; report "
                       "the nominal size from the model card alongside.",
               label="tab:experiment-compute")


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#5c5c57", "ytick.color": "#5c5c57"})
    return plt


def plot_deltas(deltas, order, datasets):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.7 * len(datasets), 3.3), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = deltas[deltas["dataset"] == ds]
        models = [m for m in order if m in set(d["model"])]
        x = np.arange(len(models))
        for j, c in enumerate(ALTERED):
            col, mk = COND_STYLE[c]
            s = d[d["comparison"] == f"real - {c}"].set_index("model").reindex(models)
            ok = s["delta"].notna().to_numpy()
            y, lo, hi = s["delta"].to_numpy(), s["ci_low"].to_numpy(), s["ci_high"].to_numpy()
            ax.errorbar((x + (j - 1) * 0.22)[ok], 100 * y[ok], yerr=[100 * (y - lo)[ok], 100 * (hi - y)[ok]],
                        fmt=mk, ms=6, color=col, mec="white", mew=1.0, elinewidth=1.4, capsize=0,
                        label=f"Correct image vs {COND_LABEL[c].lower()}")
        ax.axhline(0, color="#8a8a85", lw=1)
        ax.set_xticks(x, [MODEL_LABEL.get(m, m).replace(" (", "\n(").replace(" fusion", "\nfusion") for m in models],
                      fontsize=8)
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=9, color="#2b2b29", loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    axes[0][0].set_ylabel("Accuracy change (percentage points)\npositive = the correct image helps")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"fig_deltas.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sens_util(props, datasets):
    """Sensitivity (flip rate, correct image -> no image information) vs utility (Delta_visual)."""
    plt = setup_mpl()
    p = props[props["model"] != "q_only"].dropna(subset=["flip_none", "delta_visual"])
    if p.empty:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.3), sharex=True, sharey=True,
                             squeeze=False)
    xmax = max(0.25, 100 * p["flip_none"].max() * 1.15)
    for ax, ds in zip(axes[0], datasets):
        s = p[p["dataset"] == ds]
        ax.axhline(0, color="#8a8a85", lw=1)
        ax.axvline(100 * SENS_THRESHOLD, color="#b4b3ad", lw=1, ls=(0, (3, 3)))
        for r in s.itertuples():
            col, mk = MODEL_STYLE.get(r.model, ("#5c5c57", "o"))
            ax.errorbar(100 * r.flip_none, 100 * r.delta_visual,
                        yerr=[[100 * (r.delta_visual - r.dv_ci_low)], [100 * (r.dv_ci_high - r.delta_visual)]],
                        fmt=mk, ms=7, color=col, mec="white", mew=1.0, elinewidth=1.2, capsize=0, clip_on=False,
                        alpha=0.9, label=MODEL_LABEL.get(r.model, r.model))
        ax.set_xlim(0, xmax)
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=9, color="#2b2b29", loc="left")
        ax.set_xlabel("Visual sensitivity: answers changed after\nvisual-information neutralization (%)")
        ax.grid(color="#e6e5e0", lw=0.6)
    axes[0][0].set_ylabel("Visual utility: $\\Delta_{visual}$ (pp)")
    seen = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, ncol=len(seen), loc="upper center",
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"fig_sensitivity_utility.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
