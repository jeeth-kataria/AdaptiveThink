# ── Cell 2: Mount NVMe + create directory layout ─────────────────────────────
import os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(os.getcwd())
# Handle if running from notebooks/ subdirectory
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
    os.chdir(REPO_ROOT)
    print(f"Changed working directory to: {REPO_ROOT}")

# Add src/ to Python path for imports
sys.path.insert(0, str(REPO_ROOT / "src"))
print(f"✅ Added to Python path: {REPO_ROOT / 'src'}")

def _mount_nvme(target: str) -> bool:
    """Format + mount Azure local NVMe. Safe to call repeatedly (checks if mounted)."""
    dev = "/dev/nvme0n1"
    if not os.path.exists(dev):
        return False
    # Already mounted?
    result = subprocess.run(["mountpoint", "-q", target])
    if result.returncode == 0:
        print(f"  ℹ️  {target} already mounted.")
        return True
    try:
        subprocess.run(["mkfs.ext4", "-F", dev], check=True, capture_output=True)
        Path(target).mkdir(parents=True, exist_ok=True)
        subprocess.run(["mount", dev, target], check=True)
        subprocess.run(["chmod", "777", target], check=True)
        print(f"  ✅ NVMe mounted at {target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  NVMe mount failed ({e}) — using local fallback")
        return False

# ── Mount ────────────────────────────────────────────────────────────────────
nvme_mounted = _mount_nvme(CFG["DATA_DIR"])

# If NVMe unavailable fall back to repo-local dirs
DATA_DIR    = Path(CFG["DATA_DIR"]    if nvme_mounted else "data")
OUTPUTS_DIR = Path(CFG["OUTPUTS_DIR"] if nvme_mounted else "outputs")

# ── Subdirectories ───────────────────────────────────────────────────────────
for d in [DATA_DIR, OUTPUTS_DIR,
          REPO_ROOT / "logs",
          REPO_ROOT / "results" / "pareto",
          REPO_ROOT / "results" / "ablations",
          REPO_ROOT / "results" / "figures",
          REPO_ROOT / "results" / "onDevice"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Symlinks from repo into NVMe so all scripts use relative paths ────────────
for name, target in [("data", DATA_DIR), ("outputs", OUTPUTS_DIR)]:
    link = REPO_ROOT / name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target)
        print(f"  🔗 {link} -> {target}")
    elif link.is_symlink():
        print(f"  ℹ️  {link} -> {os.readlink(link)} (already linked)")
    else:
        print(f"  ℹ️  {link} exists as real dir (NVMe unavailable — using local)")

# Export for later cells
import builtins
builtins._AT_DATA_DIR    = DATA_DIR
builtins._AT_OUTPUTS_DIR = OUTPUTS_DIR
builtins._AT_REPO_ROOT   = REPO_ROOT

print(f"\n✅ Layout ready.  data={DATA_DIR}  outputs={OUTPUTS_DIR}")
