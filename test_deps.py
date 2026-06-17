#!/usr/bin/env python3
"""Test dependencies before full pilot run"""

import sys
import subprocess

def test_imports():
    """Test all critical imports"""
    print("=== Testing imports ===")
    
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("datasets", "Datasets"),
        ("wandb", "WandB"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    passed = []
    failed = []
    
    for module, name in tests:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✅ {name:15s} {version}")
            passed.append(name)
        except ImportError as e:
            print(f"  ❌ {name:15s} FAILED: {e}")
            failed.append(name)
    
    # Test CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        else:
            print(f"\n  ⚠️  CUDA not available")
    except Exception as e:
        print(f"\n  ❌ CUDA check failed: {e}")
    
    print(f"\n=== Summary ===")
    print(f"  Passed: {len(passed)}/{len(tests)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        return False
    return True

def test_adaptivethink():
    """Test local package imports"""
    print("\n=== Testing adaptivethink package ===")
    try:
        from adaptivethink.data.loaders import build_verifier_pool
        from adaptivethink.verifier.model import DifficultyVerifier
        from adaptivethink.router.train_grpo import main as train_main
        print("  ✅ All adaptivethink imports work")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("  → Run: pip install -e .")
        return False

if __name__ == "__main__":
    print("🧪 AdaptiveThink Dependency Test\n")
    
    import_ok = test_imports()
    package_ok = test_adaptivethink()
    
    if import_ok and package_ok:
        print("\n✅ ALL CHECKS PASSED - Ready to run pilot!")
        sys.exit(0)
    else:
        print("\n❌ SOME CHECKS FAILED - Fix dependencies first")
        sys.exit(1)
