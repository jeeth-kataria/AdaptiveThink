# ── Cell 20: GGUF Q4_K_M export ───────────────────────────────────────────────
import subprocess, sys, json
from pathlib import Path

OUTPUTS_DIR = _AT_OUTPUTS_DIR
best_adapter = str(_AT_BEST_SEED_DIR)
gguf_out = OUTPUTS_DIR / "gguf" / "router-1p5b-Q4_K_M.gguf"

if gguf_out.exists():
    print(f"✅ GGUF already exists: {gguf_out} ({gguf_out.stat().st_size/1e9:.2f} GB)")
else:
    print("=== GGUF Q4_K_M export ===")
    subprocess.run(
        [sys.executable, "src/adaptivethink/quantize/export_gguf.py",
         "--adapter", best_adapter,
         "--merged-dir", str(OUTPUTS_DIR / "router-merged"),
         "--out", str(gguf_out),
         "--quant-type", "Q4_K_M"],
        check=True,
    )
    print(f"\n✅ GGUF: {gguf_out} ({gguf_out.stat().st_size/1e9:.2f} GB)")

# ── Host-side parity check ────────────────────────────────────────────────────
print("\n=== Verifying GGUF parity (50 GSM8K items) ===")
hf_res = "results/ours_full.json"
gguf_res = "results/gguf_parity.json"

subprocess.run(
    [sys.executable, "eval/run_benchmarks.py",
     "--gguf", str(gguf_out),
     "--verifier-ckpt", str(OUTPUTS_DIR / "verifier-400m" / "best.pt"),
     "--route-mode", "threshold",
     "--benchmark", "gsm8k",
     "--n", "50",
     "--tag", "gguf_parity",
     "--out", gguf_res],
    check=True,
)

hf_acc = json.load(open(hf_res))["benchmarks"]["gsm8k"]["pass@1"]
gguf_acc = json.load(open(gguf_res))["benchmarks"]["gsm8k"]["pass@1"]
delta = abs(hf_acc - gguf_acc)

print(f"  HF:   {hf_acc:.3f}")
print(f"  GGUF: {gguf_acc:.3f}")
print(f"  Δ:    {delta:.3f}")

assert delta < 0.06, (
    f"ABORT — GGUF accuracy delta {delta:.3f} > 6% — likely quant/merge bug."
)
if delta > 0.03:
    print("  ⚠️  Delta > 3% — acceptable but investigate if > 5%.")
else:
    print("  ✅ GGUF parity confirmed.")

import builtins; builtins._AT_GGUF = str(gguf_out)
