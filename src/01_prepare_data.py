"""Step 1 — build the yes/no question sets for VQA-RAD and SLAKE (English).

    python 01_prepare_data.py --vqarad_json "PATH/VQA_RAD Dataset Public.json"   # official OSF release (auto-detected)
    --vqarad_split fixed     (default) image-disjoint split of Wu et al. 2025 (data_splits/vqa_rad_fixed/)
    --vqarad_split official  original split (train/test share images) -- sensitivity analysis only
    add --slake_dir PATH if SLAKE is already unzipped locally (else auto-download from HuggingFace)

Outputs: work/data/questions.csv, work/data/images/, work/results/data_summary.csv, work/results/data_checks.json
"""
import argparse, json, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import DATA, RES, SWAP_SEED, GRAY_RGB, pixel_hash, norm_answer

IMG_DIR = DATA / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
META_COLS = ["qtype", "modality", "region", "phrase_type", "base_type"]
SPLIT_DIR = Path(__file__).resolve().parent / "data_splits" / "vqa_rad_fixed"


def save_image(img: Image.Image, dataset: str):
    h = pixel_hash(img)
    p = IMG_DIR / dataset / f"{h}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:                       # a file left half-written by an interrupted run is rewritten
            with Image.open(p) as chk:
                chk.load()
            return str(p), h
        except Exception:
            p.unlink()
    tmp = p.with_suffix(".tmp.png")
    img.convert("RGB").save(tmp)   # atomic write: never leaves a truncated image behind
    tmp.replace(p)
    return str(p), h


def row(dataset, split, qid, img, question, answer, **meta):
    path, h = save_image(img, dataset)
    r = dict(dataset=dataset, split=split, qid=qid, image_path=path, image_hash=h,
             question=str(question).strip(), answer=answer)
    for c in META_COLS:
        r[c] = str(meta.get(c, "") or "").strip()
    return r


# ---------------------------------------------------------------- VQA-RAD
def _nq(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split())


def load_vqarad_osf(json_path: Path, split_mode: str = "fixed"):
    """Official OSF release (images + metadata).
    split_mode 'fixed'   : image-disjoint train/validation/test of Wu et al. (2025); questions are matched
                           back to the OSF records (image name + question text) to recover qid and metadata.
    split_mode 'official': original split (test = phrase_type starting with 'test'); train/test share images."""
    items = json.loads(Path(json_path).read_text())
    # Image folder name varies (e.g. Kaggle renames OSF archives): index every image found
    # next to the JSON (recursively), then one level up if needed.
    index = {}
    for root in (Path(json_path).parent, Path(json_path).parent.parent):
        for p in root.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name not in index:
                index[p.name] = p
        if all(it["image_name"] in index for it in items):
            break
    missing = sorted({it["image_name"] for it in items} - set(index))
    if missing:
        raise SystemExit(f"VQA-RAD: {len(missing)} images not found near {json_path} (e.g. {missing[:3]})")
    print(f"  VQA-RAD images found: {len(index)}")

    def osf_row(split, it):
        return row("vqarad", split, f"vqarad-{it['qid']}", Image.open(index[it["image_name"]]), it["question"],
                   it["answer"], qtype=it.get("question_type", ""), region=it.get("image_organ", ""),
                   phrase_type=str(it.get("phrase_type", "")))

    if split_mode == "official":
        rows = [osf_row("test" if str(it.get("phrase_type", "")).startswith("test") else "train", it) for it in items]
        return pd.DataFrame(rows), "OSF official release, original split (train/test share images)", {}

    # ---- image-disjoint split
    lookup = {}
    for it in items:
        lookup.setdefault((it["image_name"], _nq(it["question"])), []).append(it)
    rows, stats = [], {"matched_to_osf": 0, "unmatched": 0}
    for fname, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        entries = json.loads((SPLIT_DIR / f"{fname}_fixed_split.json").read_text())
        for i, e in enumerate(entries):
            cands = lookup.get((e["image_name"], _nq(e["question"])), [])
            same = [c for c in cands if norm_answer(c["answer"]) == norm_answer(e["answer"])]
            if cands:
                it = (same or cands)[0]
                cands.remove(it)                     # each OSF record is used at most once
                r = osf_row(split, it)
                r["answer"] = e["answer"]            # the split file's answer is authoritative
                stats["matched_to_osf"] += 1
            else:
                r = row("vqarad", split, f"vqarad-fixed-{fname}-{i}", Image.open(index[e["image_name"]]),
                        e["question"], e["answer"])
                stats["unmatched"] += 1
            rows.append(r)
    print(f"  fixed split: {stats}")
    return (pd.DataFrame(rows), "OSF images + image-disjoint split of Wu et al. (2025), Nat Commun 16:7866", stats)


def load_vqarad_hf():
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")
    rows = []
    for split in ("train", "test"):
        for i, ex in enumerate(ds[split]):
            rows.append(row("vqarad", split, f"vqarad-{split}-{i}", ex["image"], ex["question"], ex["answer"]))
    print("  WARNING: HuggingFace VQA-RAD has no metadata -> no question-type strata, swaps not stratified.")
    return pd.DataFrame(rows), "HuggingFace flaviagiammarino/vqa-rad (filtered split, no metadata)", {}


def find_vqarad_json():
    """Look for the official release in Kaggle inputs and in ./data_raw before falling back to HF."""
    for root in (Path("/kaggle/input"), Path("data_raw"), DATA.parent / "data_raw"):
        if root.exists():
            hits = sorted(root.rglob("VQA_RAD Dataset Public.json"))
            if hits:
                return hits[0]
    return None


# ---------------------------------------------------------------- SLAKE
def fetch_slake() -> Path:
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("BoKelvin/SLAKE", repo_type="dataset", local_dir=str(DATA / "raw_slake")))
    for z in root.rglob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)
    return root


def find_one(root: Path, names, want_dir=False):
    for n in names:
        for h in root.rglob(n):
            if h.is_dir() == want_dir:
                return h
    return None


def load_slake(root: Path):
    files = {"train": find_one(root, ["train.json"]),
             "val": find_one(root, ["validate.json", "validation.json", "val.json"]),
             "test": find_one(root, ["test.json"])}
    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise SystemExit(f"SLAKE: json for {missing} not found under {root}")
    img_dir = find_one(root, ["imgs"], want_dir=True)
    if img_dir is None:
        raise SystemExit(f"SLAKE: folder 'imgs' not found under {root} (unzip imgs.zip there)")
    rows = []
    for split, jf in files.items():
        for it in json.loads(jf.read_text()):
            if it.get("q_lang", "en") != "en":
                continue
            rows.append(row("slake", split, f"slake-{it['qid']}", Image.open(img_dir / it["img_name"]),
                            it["question"], it["answer"], qtype=it.get("content_type", ""),
                            modality=it.get("modality", ""), region=it.get("location", ""),
                            base_type=it.get("base_type", "")))
    return pd.DataFrame(rows), "SLAKE 1.0 official files (English subset)"


# ---------------------------------------------------------------- processing
def keep_yes_no(df):
    df = df.copy()
    df["answer"] = df["answer"].map(norm_answer)
    n0 = len(df)
    df = df[df["answer"].isin(["yes", "no"])].copy()
    df["label"] = (df["answer"] == "yes").astype(int)
    print(f"  kept {len(df)}/{n0} yes/no questions")
    return df


def make_val_split(df, frac=0.1):
    """VQA-RAD has no validation split: hold out 10 % of TRAIN images (grouped by image)."""
    if (df["split"] == "val").any():
        return df
    rng = np.random.default_rng(SWAP_SEED)
    imgs = np.sort(df.loc[df["split"] == "train", "image_hash"].unique())
    val_imgs = set(rng.choice(imgs, size=max(1, int(len(imgs) * frac)), replace=False))
    df.loc[(df["split"] == "train") & df["image_hash"].isin(val_imgs), "split"] = "val"
    return df


def add_swaps(df):
    """Swap image: another TEST image of the same dataset, same modality + region when available,
    always a different image. Drawn once (SWAP_SEED), shared by every model."""
    df = df.copy()
    df["swap_image_path"], df["swap_stratified"] = "", False
    rng = np.random.default_rng(SWAP_SEED)
    test = df[df["split"] == "test"].sort_values("qid")
    imgs = test.drop_duplicates("image_hash")[["image_path", "image_hash", "modality", "region"]]
    if len(imgs) < 2:
        raise SystemExit("Need at least 2 distinct test images to build swaps.")
    pool_all = imgs.to_numpy()
    for idx, r in test.iterrows():
        same = imgs[(imgs["modality"] == r["modality"]) & (imgs["region"] == r["region"])
                    & (imgs["image_hash"] != r["image_hash"])]
        stratified = (r["modality"] != "" or r["region"] != "") and len(same) >= 1
        pool = same.to_numpy() if stratified else pool_all[pool_all[:, 1] != r["image_hash"]]
        p = pool[rng.integers(len(pool))]
        df.at[idx, "swap_image_path"], df.at[idx, "swap_stratified"] = p[0], stratified
    return df


def presumed_dependence(df):
    """Question-level property fixed BEFORE any model is run, from dataset annotations only.
    SLAKE: knowledge-based questions (base_type 'kvqa' / content_type 'KG') vs vision-based questions.
    VQA-RAD has no such annotation -> analysed by annotated question_type only."""
    out = []
    for r in df.itertuples():
        if r.dataset == "slake":
            kg = str(r.base_type).lower() == "kvqa" or str(r.qtype).upper() == "KG"
            out.append("knowledge-based" if kg else "vision-based")
        else:
            out.append("not annotated")
    return out


def checks(df, sources):
    out = {"sources": sources, "licences": "TO VERIFY before submission: VQA-RAD (OSF), SLAKE 1.0"}
    for ds_name, g in df.groupby("dataset"):
        trainval = set(g.loc[g["split"].isin(["train", "val"]), "image_hash"])
        te = g[g["split"] == "test"]
        dup = te.assign(qn=te["question"].str.lower()).duplicated(["image_hash", "qn"], keep=False)
        out[ds_name] = {
            "test_questions": int(len(te)),
            "test_questions_whose_image_is_in_train_or_val": int(te["image_hash"].isin(trainval).sum()),
            "test_images_also_in_train_or_val": int(len(set(te["image_hash"]) & trainval)),
            "test_duplicate_image_question_pairs": int(dup.sum()),
            "test_phrase_types": te["phrase_type"].value_counts().to_dict(),
            "swap_stratified_rate": round(float(te["swap_stratified"].mean()), 3),
            "has_question_type_metadata": bool((te["qtype"] != "").any()),
            "test_presumed_image_dependence": te.get("presumed_image_dependence", pd.Series(dtype=str)).value_counts().to_dict(),
        }
    (RES / "data_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def summarise(df):
    rows = []
    for (ds_name, split), s in df.groupby(["dataset", "split"]):
        rows.append(dict(dataset=ds_name, split=split, questions=len(s), images=s["image_hash"].nunique(),
                         yes_rate=round(s["label"].mean(), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "data_summary.csv", index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slake_dir", type=Path, default=None)
    ap.add_argument("--vqarad_json", type=Path, default=None)
    ap.add_argument("--vqarad_split", choices=["fixed", "official"], default="fixed")
    args = ap.parse_args()

    print("VQA-RAD ...")
    vq = args.vqarad_json or find_vqarad_json()
    if vq:
        print(f"  using official release: {vq}")
    if vq:
        vr, src_vr, split_stats = load_vqarad_osf(vq, args.vqarad_split)
    else:
        vr, src_vr, split_stats = load_vqarad_hf()
    vr = keep_yes_no(vr)
    if not (vr["split"] == "val").any():
        vr = make_val_split(vr)
    print("SLAKE (English) ...")
    sl, src_sl = load_slake(args.slake_dir if args.slake_dir else fetch_slake())
    sl = keep_yes_no(sl)

    df = pd.concat([add_swaps(vr), add_swaps(sl)], ignore_index=True)   # swaps never cross datasets
    gray = IMG_DIR / "gray.png"
    Image.new("RGB", (512, 512), GRAY_RGB).save(gray)
    df["gray_image_path"] = str(gray)
    df["presumed_image_dependence"] = presumed_dependence(df)

    df.to_csv(DATA / "questions.csv", index=False)
    print(f"Saved {len(df)} questions -> {DATA / 'questions.csv'}")
    summarise(df)
    checks(df, {"vqarad": src_vr, "slake": src_sl, "vqarad_split_matching": split_stats})


if __name__ == "__main__":
    main()
