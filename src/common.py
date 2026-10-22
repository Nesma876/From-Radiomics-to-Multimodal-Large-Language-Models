"""Shared settings and helpers. Works on Kaggle and on a local GPU machine."""
import os, json, random, re, hashlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
# On Kaggle, outputs must live under /kaggle/working. Elsewhere: ./work
_default = "/kaggle/working/work" if Path("/kaggle/working").exists() else "work"
WORK = Path(os.environ.get("EXPE_WORK", _default)).resolve()
DATA = WORK / "data"          # prepared questions + images
FEAT = WORK / "features"      # cached BiomedCLIP features
PRED = WORK / "predictions"   # one CSV per model run
RES = WORK / "results"        # tables, figures, LaTeX
for d in (DATA, FEAT, PRED, RES):
    d.mkdir(parents=True, exist_ok=True)

ENV_FILE = WORK / "env.json"  # written by 00_check_env.py

# ---------------------------------------------------------------- fixed settings (pre-specified)
SEEDS = [0, 1, 2, 3, 4]
SWAP_SEED = 2026
BOOT_SEED = 2026
N_BOOT = 2000
GRAY_RGB = (128, 128, 128)
CONDITIONS = ["real", "none", "gray", "swap"]   # see ANALYSIS_PLAN.md
PROMPT = "{question}\nAnswer with yes or no only."
MAX_NEW_TOKENS = 8
N_WARMUP = 5
LAT_N = 50        # fixed test questions timed per repetition
LAT_REPS = 3
TEXT_ANSWERABLE_MIN = 4   # q_only correct in >= 4 of 5 seeds -> text-answerable

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASSICAL_HP = dict(lr=1e-4, weight_decay=1e-2, epochs=40, batch_size=32,
                    hidden=256, dropout=0.2, patience=8, attn_heads=8, max_q_tokens=64)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def pixel_hash(img) -> str:
    """md5 of decoded RGB pixels: detects identical images saved under different names."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def norm_answer(a) -> str:
    return re.sub(r"[^a-z]", "", str(a).lower())


def parse_yes_no(text: str) -> str:
    """First word of the generated text if it is yes/no, else 'unparsed' (scored wrong)."""
    words = re.findall(r"[a-z]+", str(text).lower())
    if words and words[0] in ("yes", "no"):
        return words[0]
    return "unparsed"


def load_env() -> dict:
    if ENV_FILE.exists():
        return json.loads(ENV_FILE.read_text())
    return {}
