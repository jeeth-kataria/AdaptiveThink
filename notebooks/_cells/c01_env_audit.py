# ── Cell 1: Environment audit ────────────────────────────────────────────────
import subprocess, sys, os

def _check(label, ok, detail=""):
    icon = "✅" if ok else "❌"
    print(f"  {icon} {label}" + (f"  ({detail})" if detail else ""))
    return ok

print("=== Environment Audit ===")
import torch

cuda_ok   = _check("CUDA available",      torch.cuda.is_available())
if cuda_ok:
    gpu_name  = torch.cuda.get_device_name(0)
    vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_ver  = torch.version.cuda or "0"
    drv_ver   = subprocess.run(["nvidia-smi","--query-gpu=driver_version",
                                "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
    vram_ok   = _check("VRAM ≥ 14 GB",    vram_gb >= 14,  f"{vram_gb:.1f} GB — {gpu_name}")
    cuda12_ok = _check("CUDA ≥ 11.8",     float(cuda_ver) >= 11.8, f"CUDA {cuda_ver}, driver {drv_ver}")
    a100_mode = vram_gb >= 60
    print(f"  ℹ️  Profile: {'A100 (vLLM colocated)' if a100_mode else 'T4 (HF generation)'}")
else:
    vram_ok = cuda12_ok = False
    print("  ⚠️  No GPU detected — CPU/debug mode only")

nvme = os.path.exists("/dev/nvme0n1")
_check("Azure NVMe present", nvme, "/dev/nvme0n1" if nvme else "will use ./data fallback")

py_ok = _check("Python ≥ 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

hard_failures = []
if not cuda_ok:  hard_failures.append("No CUDA")
if cuda_ok and not vram_ok: hard_failures.append(f"VRAM {vram_gb:.1f} GB < 14 GB")
if cuda_ok and not cuda12_ok: hard_failures.append(f"CUDA {cuda_ver} < 11.8")

assert not hard_failures, f"ABORT — environment does not meet minimum spec: {hard_failures}"
print("\n✅ Environment audit passed.")

# Export for later cells
import builtins
builtins._AT_A100_MODE = a100_mode
builtins._AT_VRAM_GB   = vram_gb if cuda_ok else 0.0
