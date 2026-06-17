# ── Cell 14: GRPO seed 1 ──────────────────────────────────────────────────────
if _AT_PARETO_CASE in ("C", "D"):
    raise RuntimeError(
        f"CASE {_AT_PARETO_CASE} detected in Cell 13 — fix seed-0 before running seeds 1+2."
    )
_seed1_dir = _run_grpo(1)
import builtins; builtins._AT_SEED1_DIR = _seed1_dir
