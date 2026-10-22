"""Step 3 — generative MLLM, zero-shot, same questions and conditions as step 2.

    python 03_mllm.py --backend medgemma        # google/medgemma-4b-it (plain transformers) — primary
    python 03_mllm.py --backend llava_med       # microsoft/llava-med-v1.5-mistral-7b (needs LLaVA-Med repo) — optional

Conditions: real, none (TEXT-ONLY prompt, no image token), gray, swap.
LLaVA-Med requires an image, so its 'none' condition is not run (reported as such).

Outputs: work/predictions/mllm_<backend>.csv, work/results/compute_mllm_<backend>.csv
Resumable: answers already saved are skipped if the script is restarted.
"""
import argparse, time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import (CONDITIONS, DATA, LAT_N, LAT_REPS, MAX_NEW_TOKENS, N_WARMUP, PRED, PROMPT, RES,
                    load_env, parse_yes_no, seed_everything)

ENV = load_env()
DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(ENV.get("dtype"), torch.float32)
FOUR_BIT = ENV.get("mllm_4bit", True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ======================================================================= backends
class MedGemma:
    model_id = "google/medgemma-4b-it"
    supports_text_only = True

    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        kw = dict(torch_dtype=DTYPE, device_map="auto")
        if FOUR_BIT:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                                                           bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def answer(self, image, question: str) -> str:
        content = [] if image is None else [{"type": "image", "image": image.convert("RGB")}]
        content.append({"type": "text", "text": PROMPT.format(question=question)})
        inp = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                            tokenize=True, return_dict=True, return_tensors="pt"
                                            ).to(self.model.device)
        n = inp["input_ids"].shape[-1]
        out = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.proc.decode(out[0][n:], skip_special_tokens=True)


class LlavaMed:
    model_id = "microsoft/llava-med-v1.5-mistral-7b"
    conv_mode = "mistral_instruct"   # as in the LLaVA-Med v1.5 README; change with --conv_mode if needed
    supports_text_only = False

    def __init__(self, conv_mode=None):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        self.conv_mode = conv_mode or self.conv_mode
        name = get_model_name_from_path(self.model_id)
        self.tok, self.model, self.img_proc, _ = load_pretrained_model(
            self.model_id, None, name, load_4bit=FOUR_BIT, load_8bit=False, device="cuda")
        self.model.eval()

    @torch.no_grad()
    def answer(self, image, question: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT.format(question=question))
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).cuda()
        img = process_images([image.convert("RGB")], self.img_proc, self.model.config).to(
            self.model.device, dtype=self.model.dtype)
        out = self.model.generate(ids, images=img, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        # Depending on the LLaVA version, generate() returns only new tokens or prompt + new tokens.
        new = out[0][ids.shape[1]:] if out.shape[1] > ids.shape[1] else out[0]
        return self.tok.decode(new, skip_special_tokens=True)


BACKENDS = {"medgemma": MedGemma, "llava_med": LlavaMed}


# ======================================================================= compute
def measure_latency(model, te):
    """Standardized: N_WARMUP warm-up questions, then LAT_N fixed real-image questions x LAT_REPS."""
    rows = te.sort_values("qid").head(N_WARMUP + LAT_N)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lat = []
    for rep in range(LAT_REPS):
        for k, (_, r) in enumerate(rows.iterrows()):
            img = Image.open(r["image_path"])
            sync(); t0 = time.perf_counter()
            model.answer(img, r["question"])
            sync()
            if rep > 0 or k >= N_WARMUP:
                lat.append((time.perf_counter() - t0) * 1000)
    return lat


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--conv_mode", default=None, help="llava_med only")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N test questions per dataset")
    args = ap.parse_args()
    seed_everything(0)

    df = pd.read_csv(DATA / "questions.csv", keep_default_na=False)
    te = df[df["split"] == "test"].sort_values(["dataset", "qid"]).reset_index(drop=True)
    if args.limit:
        te = te.groupby("dataset").head(args.limit).reset_index(drop=True)
    img_col = {"real": "image_path", "gray": "gray_image_path", "swap": "swap_image_path", "none": None}

    out_f = PRED / f"mllm_{args.backend}.csv"
    done = set()
    if out_f.exists():
        prev = pd.read_csv(out_f)
        done = set(zip(prev["qid"], prev["condition"]))
        print(f"Resuming: {len(done)} answers already saved")

    model = LlavaMed(args.conv_mode) if args.backend == "llava_med" else BACKENDS[args.backend]()
    conditions = [c for c in CONDITIONS if c != "none" or model.supports_text_only]
    if "none" not in conditions:
        print("NOTE: this model requires an image; condition 'none' is not run.")

    buffer, n_new = [], 0
    for cond in conditions:
        for _, r in te.iterrows():
            if (r["qid"], cond) in done:
                continue
            img = None if img_col[cond] is None else Image.open(r[img_col[cond]])
            raw = model.answer(img, r["question"])
            ans = parse_yes_no(raw)
            buffer.append(dict(dataset=r["dataset"], model=f"mllm_{args.backend}", seed=-1, condition=cond,
                               qid=r["qid"], label=r["label"], raw_output=raw, answer=ans,
                               pred=1 if ans == "yes" else (0 if ans == "no" else -1)))
            n_new += 1
            if len(buffer) >= 50:
                pd.DataFrame(buffer).to_csv(out_f, mode="a", header=not out_f.exists(), index=False)
                buffer = []
                print(f"  {cond}: {n_new} new answers saved", flush=True)
    if buffer:
        pd.DataFrame(buffer).to_csv(out_f, mode="a", header=not out_f.exists(), index=False)
    print(f"Saved -> {out_f}")

    comp_f = RES / f"compute_mllm_{args.backend}.csv"
    if not comp_f.exists() and not args.limit:
        print("Measuring latency (standardized protocol) ...")
        lat = measure_latency(model, te)
        n_params = sum(p.numel() for p in model.model.parameters())
        pd.DataFrame([dict(model=f"mllm_{args.backend}", hf_id=type(model).model_id, total_params=n_params,
                           trainable_params=0, train_seconds_median_per_seed=0.0,
                           latency_ms_median=float(np.median(lat)), latency_ms_q1=float(np.percentile(lat, 25)),
                           latency_ms_q3=float(np.percentile(lat, 75)), n_timed=len(lat),
                           peak_mem_mb=(torch.cuda.max_memory_allocated() / 1024**2
                                        if torch.cuda.is_available() else float("nan")),
                           gpu=ENV.get("gpu", ""), precision=(f"4-bit nf4, {DTYPE} compute" if FOUR_BIT else str(DTYPE)),
                           note="parameter count of the loaded (possibly quantized) model")]
                     ).to_csv(comp_f, index=False)
        print(f"Saved -> {comp_f}")


if __name__ == "__main__":
    main()
