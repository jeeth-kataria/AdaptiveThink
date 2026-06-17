# ── Cell 21: On-device benchmark (Day 8+ — stub) ─────────────────────────────
if not CFG["RUN_ONDEVICE"]:
    print("⏭️  RUN_ONDEVICE=False — skipping.")
    print("    Set CFG['RUN_ONDEVICE']=True in Cell 0 to enable.")
    print("    Requires: Galaxy connected via ADB + scripts/08_galaxy_bench.sh")
else:
    print("=== On-device benchmark ===")
    print("⚠️  This cell is a STUB — implementation pending Day 8.")
    print("\nManual steps:")
    print("  1. Connect Galaxy via USB, enable USB debugging")
    print("  2. Verify: adb devices")
    print("  3. Push GGUF:")
    print(f"     adb push {_AT_GGUF} /sdcard/adaptivethink/")
    print("  4. Run battery harness:")
    print("     bash scripts/08_galaxy_bench.sh")
    print("  5. Pull results:")
    print("     adb pull /sdcard/adaptivethink/results results/onDevice/")
    print("\n📌 Results will be in results/onDevice/galaxy_pareto.csv")
