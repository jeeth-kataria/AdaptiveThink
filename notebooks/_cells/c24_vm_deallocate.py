# ── Cell 24: Deallocate Azure VM (optional — saves cost) ─────────────────────
import subprocess, time, os

if not CFG["AUTO_DEALLOCATE"]:
    print("⏭️  AUTO_DEALLOCATE=False — VM will keep running.")
    print("    MANUAL: az vm deallocate -g <rg> -n <vm> --no-wait")
else:
    print("=== Azure VM auto-deallocate ===")
    print("⚠️  This will SHUT DOWN the VM in 60 seconds.")
    print("    Press Ctrl+C in the next 60 seconds to cancel.\n")
    
    for i in range(60, 0, -10):
        print(f"    Deallocating in {i}s...")
        time.sleep(10)
    
    sub_id = CFG["AZURE_SUBSCRIPTION_ID"]
    rg     = CFG["AZURE_RESOURCE_GROUP"]
    vm     = CFG["AZURE_VM_NAME"]
    
    cmd = ["az", "vm", "deallocate",
           "--subscription", sub_id,
           "--resource-group", rg,
           "--name", vm,
           "--no-wait"]
    
    print(f"\n🔌 Deallocating {vm} in {rg}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Deallocation command sent.")
        print(f"   Check status: az vm show -g {rg} -n {vm} --query powerState")
        print("   The VM will stop billing within ~2 minutes.")
    else:
        print(f"❌ Deallocation failed:\n{result.stderr}")
        print("   MANUAL: run the command above in your terminal.")
