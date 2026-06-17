#!/usr/bin/env python3
"""Assemble AdaptiveThink_Train.ipynb from individual cell files."""
import json
from pathlib import Path

CELLS_DIR = Path("notebooks/_cells")
SECTIONS = [
    ("# AdaptiveThink — Azure Full Training Pipeline\n\n"
     "**Run this notebook top-to-bottom on Azure NC24ads_A100_v4.**\n\n"
     "- Estimated cost: ~$85 (spot) for full 3-seed run\n"
     "- Estimated time: ~120 GPU hours total\n"
     "- Only Cell 0 should ever be edited"),
    
    ("## Section 0 — Config", ["c00_config.py"]),
    ("## Section 1 — Setup", ["c01_env_audit.py", "c02_nvme_mount.py",
                               "c03_secrets.py", "c04_deps.py", "c05_tests.py"]),
    ("## Section 2 — Data", ["c06_data_download.py", "c07_teacher_labels.py",
                              "c08_grpo_data.py"]),
    ("## Section 3 — Verifier (Stage 1)", ["c09_verifier_train.py",
                                            "c10_hf_push_verifier.py"]),
    ("## Section 4 — GRPO Router (Stage 2)", ["c11_smoke_test.py", "c12_grpo_seed0.py",
                                               "c13_pareto_check.py", "c14_grpo_seed1.py",
                                               "c15_grpo_seed2.py", "c16_3seed_ci.py"]),
    ("## Section 5 — TTRL (optional)", ["c17_ttrl.py"]),
    ("## Section 6 — Eval", ["c18_full_eval.py", "c19_pareto_chart.py"]),
    ("## Section 7 — Quantise", ["c20_gguf_export.py"]),
    ("## Section 8 — On-device (Day 8+)", ["c21_ondevice_stub.py"]),
    ("## Section 9 — Upload + Wrap-up", ["c22_hf_upload.py", "c23_progress_docs.py",
                                          "c24_vm_deallocate.py", "c24b_git_backup.py"]),
]

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source]
    }

cells = []

for item in SECTIONS:
    if isinstance(item, str):
        # Top-level markdown
        cells.append(make_markdown_cell(item))
    else:
        # Section header + code cells
        section_title, cell_files = item
        cells.append(make_markdown_cell(section_title))
        for fname in cell_files:
            code = (CELLS_DIR / fname).read_text()
            cells.append(make_code_cell(code))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path("notebooks/AdaptiveThink_Train.ipynb")
out_path.write_text(json.dumps(notebook, indent=1))
print(f"✅ Notebook written: {out_path}")
print(f"   {len(cells)} cells total")
