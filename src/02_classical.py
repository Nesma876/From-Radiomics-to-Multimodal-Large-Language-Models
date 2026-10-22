"""Step 2 — classical fusion baselines on a frozen BiomedCLIP encoder.

    python 02_classical.py

Models: majority, q_only, early, intermediate (co-attention); 5 seeds each.
Conditions at test time: real, none (training-set mean image features), gray, swap.
Outputs: work/predictions/classical.csv, work/results/compute_classical.csv
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common import (BIOMEDCLIP, CLASSICAL_HP as HP, CONDITIONS, DATA, FEAT, PRED, RES,
                    SEEDS, N_WARMUP, LAT_N, LAT_REPS, seed_everything, load_env)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================= encoder
class FrozenBiomedCLIP:
    def __init__(self):
        import open_clip
        self.model, self.preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP)
        self.model.eval().to(DEV)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def images(self, pil_list):
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_list]).to(DEV)
        emb = self.model.encode_image(x, normalize=True)                 # [B, 512]
        tok = self.model.visual.trunk.forward_features(x)                # [B, 197, 768]
        return emb.float().cpu(), tok.half().cpu()

    @torch.no_grad()
    def texts(self, str_list):
        ids = self.tokenizer(str_list).to(DEV)                           # [B, 256]
        emb = self.model.encode_text(ids, normalize=True)                # [B, 512]
        ids = ids[:, :HP["max_q_tokens"]]
        mask = ids != 0                                                  # PubMedBERT pad id = 0
        tok = self.model.text.transformer(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return emb.float().cpu(), tok.half().cpu(), mask.cpu()


def cache_features(df):
    f = FEAT / "biomedclip.pt"
    if f.exists():
        return torch.load(f)
    enc = FrozenBiomedCLIP()
    paths = sorted(set(df["image_path"]) | set(df["gray_image_path"])
                   | set(df["swap_image_path"].dropna()) - {""})
    img = {}
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        e, t = enc.images([Image.open(p) for p in batch])
        for k, p in enumerate(batch):
            img[p] = (e[k], t[k])
    qs = {}
    rows = df[["qid", "question"]].to_numpy()
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        e, t, m = enc.texts([q for _, q in batch])
        for k, (qid, _) in enumerate(batch):
            qs[qid] = (e[k], t[k], m[k])
    feats = {"img": img, "q": qs}
    torch.save(feats, f)
    return feats


def add_mean_image(feats, df):
    """'none' condition for classical models: mean image embedding and patch tokens over the
    distinct TRAINING images of each dataset (in-distribution, no image-specific information)."""
    feats["mean"] = {}
    for ds, g in df[df["split"] == "train"].groupby("dataset"):
        paths = g.drop_duplicates("image_hash")["image_path"]
        emb = torch.stack([feats["img"][p][0] for p in paths]).mean(0)
        tok = torch.stack([feats["img"][p][1].float() for p in paths]).mean(0)
        feats["mean"][ds] = (emb, tok)
    return feats


# ======================================================================= heads
def mlp(d_in, hidden, dropout):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class QOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = mlp(512, HP["hidden"], HP["dropout"])

    def forward(self, b):
        return self.head(b["q_emb"]).squeeze(-1)


class Early(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = mlp(1024, HP["hidden"], HP["dropout"])

    def forward(self, b):
        return self.head(torch.cat([b["img_emb"], b["q_emb"]], -1)).squeeze(-1)


class CoAttention(nn.Module):
    """Bidirectional co-attention between image patch tokens and question tokens."""
    def __init__(self, d_tok=768):
        super().__init__()
        h = HP["hidden"]
        self.pi, self.pq = nn.Linear(d_tok, h), nn.Linear(d_tok, h)
        self.q2i = nn.MultiheadAttention(h, HP["attn_heads"], dropout=HP["dropout"], batch_first=True)
        self.i2q = nn.MultiheadAttention(h, HP["attn_heads"], dropout=HP["dropout"], batch_first=True)
        self.ln_q, self.ln_i = nn.LayerNorm(h), nn.LayerNorm(h)
        self.head = mlp(2 * h, h, HP["dropout"])

    def forward(self, b):
        i, q, m = self.pi(b["img_tok"]), self.pq(b["q_tok"]), b["q_mask"]
        q = self.ln_q(q + self.q2i(q, i, i, need_weights=False)[0])
        i = self.ln_i(i + self.i2q(i, q, q, key_padding_mask=~m, need_weights=False)[0])
        mf = m.unsqueeze(-1).float()
        q_pool = (q * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.head(torch.cat([q_pool, i.mean(1)], -1)).squeeze(-1)


MODELS = {"q_only": QOnly, "early": Early, "intermediate": CoAttention}


# ======================================================================= batching
def make_batch(feats, qids, img_paths, labels=None):
    """img_paths entries starting with 'MEAN:' use the dataset's mean training-image features."""
    q = [feats["q"][x] for x in qids]
    im = [feats["mean"][p[5:]] if p.startswith("MEAN:") else feats["img"][p] for p in img_paths]
    b = {"q_emb": torch.stack([x[0] for x in q]), "q_tok": torch.stack([x[1] for x in q]).float(),
         "q_mask": torch.stack([x[2] for x in q]),
         "img_emb": torch.stack([x[0] for x in im]), "img_tok": torch.stack([x[1] for x in im]).float()}
    b = {k: v.to(DEV) for k, v in b.items()}
    if labels is not None:
        b["y"] = torch.tensor(labels, dtype=torch.float32, device=DEV)
    return b


@torch.no_grad()
def predict_probs(model, feats, df, img_col):
    model.eval()
    out = []
    for i in range(0, len(df), 256):
        s = df.iloc[i:i + 256]
        out.append(torch.sigmoid(model(make_batch(feats, s["qid"].tolist(), s[img_col].tolist()))).cpu())
    return torch.cat(out).numpy()


def train_one(name, feats, tr, va, seed):
    seed_everything(seed)
    model = MODELS[name]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    best, best_state, wait, epochs = -1.0, None, 0, 0
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(HP["epochs"]):
        epochs += 1
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), HP["batch_size"]):
            s = tr.iloc[order[i:i + HP["batch_size"]]]
            b = make_batch(feats, s["qid"].tolist(), s["image_path"].tolist(), s["label"].tolist())
            loss = F.binary_cross_entropy_with_logits(model(b), b["y"])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = ((predict_probs(model, feats, va, "image_path") >= 0.5) == va["label"].to_numpy()).mean()
        if acc > best:
            best, wait = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= HP["patience"]:
                break
    train_s = time.perf_counter() - t0   # head training only; feature extraction is cached once
    model.load_state_dict(best_state)
    return model, best, train_s, epochs


# ======================================================================= compute measurement
def measure_compute(name, model, enc, te):
    """End-to-end latency at batch size 1 (image preprocessing + encoding + head), standardized:
    N_WARMUP warm-up questions, then LAT_N fixed questions repeated LAT_REPS times."""
    if DEV != "cuda":
        return {}
    model.eval()
    rows = te.sort_values("qid").head(N_WARMUP + LAT_N)
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for rep in range(LAT_REPS):
        for k, (_, r) in enumerate(rows.iterrows()):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ie, it = enc.images([Image.open(r["image_path"])])
            qe, qt, qm = enc.texts([r["question"]])
            b = {"q_emb": qe, "q_tok": qt.float(), "q_mask": qm, "img_emb": ie, "img_tok": it.float()}
            with torch.no_grad():
                model({k2: v.to(DEV) for k2, v in b.items()})
            torch.cuda.synchronize()
            if rep > 0 or k >= N_WARMUP:
                lat.append((time.perf_counter() - t0) * 1000)
    head = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in enc.model.parameters())
    return dict(model=name, total_params=head + encoder, trainable_params=head,
                latency_ms_median=float(np.median(lat)), latency_ms_q1=float(np.percentile(lat, 25)),
                latency_ms_q3=float(np.percentile(lat, 75)), n_timed=len(lat),
                peak_mem_mb=torch.cuda.max_memory_allocated() / 1024**2,
                gpu=load_env().get("gpu", ""), precision="fp32 head, frozen BiomedCLIP")


# ======================================================================= main
def main():
    df = pd.read_csv(DATA / "questions.csv", keep_default_na=False)
    feats = add_mean_image(cache_features(df), df)
    enc = FrozenBiomedCLIP() if DEV == "cuda" else None
    preds, compute, train_log = [], [], []
    cols = ["dataset", "model", "seed", "condition", "qid", "label", "prob", "pred"]
    first_ds = sorted(df["dataset"].unique())[0]

    for ds_name, g in df.groupby("dataset"):
        tr, va, te = (g[g["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
        te = te.assign(none_image="MEAN:" + ds_name)
        img_col = {"real": "image_path", "none": "none_image", "gray": "gray_image_path", "swap": "swap_image_path"}
        print(f"\n== {ds_name}: train {len(tr)}  val {len(va)}  test {len(te)}")

        maj = int(tr["label"].mean() >= 0.5)
        for cond in CONDITIONS:
            preds.append(te.assign(model="majority", seed=-1, condition=cond, prob=float(maj), pred=maj)[cols])

        for name in MODELS:
            for seed in SEEDS:
                model, vacc, train_s, epochs = train_one(name, feats, tr, va, seed)
                train_log.append(dict(dataset=ds_name, model=name, seed=seed, val_acc=vacc,
                                      train_seconds=train_s, epochs=epochs))
                for cond in CONDITIONS:
                    p = predict_probs(model, feats, te, img_col[cond])
                    preds.append(te.assign(model=name, seed=seed, condition=cond, prob=p,
                                           pred=(p >= 0.5).astype(int))[cols])
                print(f"  {name:12s} seed {seed}: val acc {vacc:.3f}  ({train_s:.1f} s, {epochs} epochs)")
                if seed == SEEDS[0] and ds_name == first_ds and enc is not None:
                    compute.append(measure_compute(name, model, enc, te))

    pd.concat(preds).to_csv(PRED / "classical.csv", index=False)
    pd.DataFrame(train_log).to_csv(RES / "train_log_classical.csv", index=False)
    if compute:
        c = pd.DataFrame(compute)
        tl = pd.DataFrame(train_log).groupby("model")["train_seconds"].median().rename("train_seconds_median_per_seed")
        c.merge(tl, left_on="model", right_index=True).to_csv(RES / "compute_classical.csv", index=False)
    print(f"\nSaved predictions -> {PRED / 'classical.csv'}")


if __name__ == "__main__":
    main()
