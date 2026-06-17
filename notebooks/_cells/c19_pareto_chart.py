# ── Cell 19: Pareto chart + KPI table ────────────────────────────────────────
import subprocess, sys
from pathlib import Path
from IPython.display import Image, display, Markdown

runs = [f for f in Path("results").glob("*.json")
        if f.stem not in ("baseline",) and f.stem != "baseline_quick"]
runs_str = " ".join(str(r) for r in runs)

cmd = [sys.executable, "eval/plots.py",
       "--baseline", "results/baseline.json",
       "--runs"] + [str(r) for r in runs] + \
      ["--outdir", "results/figures"]

subprocess.run(cmd, check=True)

# ── Display KPI table inline ──────────────────────────────────────────────────
kpi_md = Path("results/figures/kpi_table.md").read_text()
display(Markdown(kpi_md))

# ── Display Pareto charts ─────────────────────────────────────────────────────
for png in sorted(Path("results/figures").glob("pareto_*.png")):
    print(f"\n--- {png.stem} ---")
    display(Image(filename=str(png)))

print("\n✅ Pareto charts + KPI table saved to results/figures/")
