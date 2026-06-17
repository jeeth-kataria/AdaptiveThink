# ── Cell 3: Secrets injection ─────────────────────────────────────────────────
import os
from pathlib import Path

_secrets = {}

# Try Azure Key Vault first (Managed Identity — no credentials needed on the VM)
if CFG["AZURE_KV_URL"]:
    try:
        from azure.identity import ManagedIdentityCredential
        from azure.keyvault.secrets import SecretClient
        _kv = SecretClient(CFG["AZURE_KV_URL"], ManagedIdentityCredential())
        _kv_map = {
            "DEEPINFRA-API-KEY": "DEEPINFRA_API_KEY",
            "WANDB-API-KEY":     "WANDB_API_KEY",
            "HF-TOKEN":          "HF_TOKEN",
        }
        for kv_name, env_name in _kv_map.items():
            try:
                _secrets[env_name] = _kv.get_secret(kv_name).value
                print(f"  ✅ {env_name} loaded from Key Vault")
            except Exception as e:
                print(f"  ⚠️  Key Vault miss for {kv_name}: {e}")
    except ImportError:
        print("  ⚠️  azure-identity not installed — skipping Key Vault")
    except Exception as e:
        print(f"  ⚠️  Key Vault unavailable: {e}")

# Fill remaining from CFG inline values
for key in ["DEEPINFRA_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", 
            "WANDB_API_KEY", "HF_TOKEN",
            "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_VM_NAME"]:
    if key not in _secrets and CFG.get(key):
        _secrets[key] = CFG[key]

# Apply to environment
for k, v in _secrets.items():
    os.environ[k] = v

# Write .env (git-ignored) for subprocesses
env_path = Path(".env")
env_path.write_text("\n".join(f"{k}={v}" for k, v in _secrets.items()) + "\n")
print(f"  📄 .env written ({len(_secrets)} keys)")

# Hard-abort if any required key is still missing
# Conditionally require API keys based on provider
if CFG.get("TEACHER_PROVIDER") == "groq":
    _required = ["GROQ_API_KEY_1", "WANDB_API_KEY", "HF_TOKEN"]
else:
    _required = ["DEEPINFRA_API_KEY", "WANDB_API_KEY", "HF_TOKEN"]

_still_missing = [k for k in _required if not os.environ.get(k)]
assert not _still_missing, (
    f"ABORT — required secrets not found in Key Vault or CFG: {_still_missing}"
)
print("\n✅ All required secrets injected.")
