# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 0 — CONFIG  (the ONLY cell you ever edit)             ║
# ╚══════════════════════════════════════════════════════════════╝

CFG = {
    # ── Azure / secrets ──────────────────────────────────────────
    # Leave blank to fall back to Azure Key Vault Managed Identity
    "DEEPINFRA_API_KEY": "",
    "WANDB_API_KEY":     "",
    "HF_TOKEN":          "",
    # Azure self-shutdown (fill in for auto-deallocate at end)
    "AZURE_SUBSCRIPTION_ID": "",
    "AZURE_RESOURCE_GROUP":  "",
    "AZURE_VM_NAME":         "",
    "AZURE_KV_URL":          "",  # e.g. "https://adaptivethink-kv.vault.azure.net/"

    # ── HF Hub ───────────────────────────────────────────────────
    "HF_ORG":              "statezero",
    "VERIFIER_REPO":       "statezero/verifier-400m",
    "LABELS_REPO":         "statezero/difficulty-labels",

    # ── Wandb ────────────────────────────────────────────────────
    "WANDB_PROJECT":       "adaptivethink",

    # ── Data paths ───────────────────────────────────────────────
    "DATA_DIR":            "/mnt/data",   # NVMe; falls back to ./data
    "OUTPUTS_DIR":         "/mnt/outputs",

    # ── Teacher labels ───────────────────────────────────────────
    "TEACHER_PROVIDER":    "deepinfra",   # deepinfra | openai | together
    "TEACHER_MAX_COST_USD": 50.0,

    # ── Verifier (Stage 1) ───────────────────────────────────────
    "VERIFIER_EPOCHS":     3,
    "VERIFIER_BATCH":      32,
    "VERIFIER_LR":         2e-5,
    "VERIFIER_MIN_RHO":    0.5,   # hard abort below this
    "VERIFIER_WARN_RHO":   0.7,   # warn below this

    # ── GRPO router (Stage 2) ────────────────────────────────────
    "GRPO_SEEDS":          [0, 1, 2],
    "GRPO_STEPS":          1500,   # 0 = auto by VRAM
    "GRPO_LR":             1e-5,
    "GRPO_KL_BETA":        5e-3,
    "GRPO_LAMBDA_TOK":     3e-3,
    "GRPO_LAMBDA_OBEY":    0.05,
    "GRPO_GROUP_SIZE":     8,      # 0 = auto
    "GRPO_MAX_SEQ_LEN":    2048,   # 0 = auto

    # ── TTRL (optional Stage 2.5) ────────────────────────────────
    "RUN_TTRL":            False,
    "TTRL_STEPS":          300,
    "TTRL_N_ITEMS":        500,

    # ── Eval ─────────────────────────────────────────────────────
    "EVAL_N_PER_BENCH":    200,    # items per benchmark (None = full)

    # ── On-device (Day 8+) ───────────────────────────────────────
    "RUN_ONDEVICE":        False,

    # ── Azure VM auto-deallocate at end ──────────────────────────
    "AUTO_DEALLOCATE":     False,
}

# ── Validation ───────────────────────────────────────────────────────────────
_required_if_no_kv = ["DEEPINFRA_API_KEY", "WANDB_API_KEY", "HF_TOKEN"]
_has_kv = bool(CFG["AZURE_KV_URL"])
if not _has_kv:
    _missing = [k for k in _required_if_no_kv if not CFG[k]]
    assert not _missing, (
        f"Fill in these keys in CFG (or set AZURE_KV_URL for Key Vault): {_missing}"
    )

if CFG["AUTO_DEALLOCATE"]:
    _az = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_VM_NAME"]
    _missing_az = [k for k in _az if not CFG[k]]
    assert not _missing_az, f"AUTO_DEALLOCATE=True but missing: {_missing_az}"

print("✅ Config validated.")
print(f"   Seeds: {CFG['GRPO_SEEDS']}  |  λ_tok={CFG['GRPO_LAMBDA_TOK']}  "
      f"|  steps={CFG['GRPO_STEPS']}  |  TTRL={CFG['RUN_TTRL']}")
