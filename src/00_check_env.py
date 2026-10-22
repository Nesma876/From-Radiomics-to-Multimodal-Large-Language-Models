"""Step 0 — check the machine and choose settings. Run first; send the printed summary back.

    python 00_check_env.py
"""
import json, platform, sys
from common import ENV_FILE, WORK

info = {"python": sys.version.split()[0], "platform": platform.platform()}

try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_count"] = torch.cuda.device_count()
        info["vram_gb"] = round(p.total_memory / 1024**3, 1)
        info["compute_capability"] = f"{p.major}.{p.minor}"
        # torch reports bf16 as "supported" on T4/V100 through slow emulation; require native support (>= 8.0).
        info["bf16_native"] = p.major >= 8
except ImportError:
    info["torch"] = None
    info["cuda_available"] = False

for pkg in ["transformers", "open_clip", "datasets", "bitsandbytes", "accelerate", "scipy", "pandas"]:
    try:
        mod = __import__(pkg)
        info[pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        info[pkg] = None

vram = info.get("vram_gb", 0) or 0
# Gemma-3-based models (MedGemma) are numerically unstable in float16. GPUs without bfloat16
# (Kaggle T4 / P100, V100) therefore compute in float32. MedGemma-4B needs ~9 GB in bf16, ~17 GB in fp32.
bf16 = bool(info.get("bf16_native"))
info["dtype"] = "bfloat16" if bf16 else "float32"
info["mllm_4bit"] = vram < (12 if bf16 else 22)

ENV_FILE.write_text(json.dumps(info, indent=2))

print("=" * 60)
for k, v in info.items():
    print(f"{k:16s}: {v}")
print("=" * 60)
if not info["cuda_available"]:
    print("WARNING: no CUDA GPU detected. Steps 02 and 03 need a GPU.")
elif vram < 14:
    print("WARNING: less than 15 GB of GPU memory. MedGemma-4B in 4-bit is the safer MLLM choice.")
print(f"MLLM weights: {'4-bit nf4' if info['mllm_4bit'] else info['dtype']}; compute dtype: {info['dtype']}.")
print(f"Outputs go to: {WORK}")
