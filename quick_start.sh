#!/usr/bin/env bash
# Quick Start - Run this after git clone
# This installs all dependencies and prepares the notebook to run

set -e

echo "🚀 AdaptiveThink Quick Setup"
echo "================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "✓ GPU detected: $GPU ($VRAM MB)"
else
    echo "⚠ No GPU detected (CPU mode)"
fi

echo ""
echo "📦 Installing dependencies..."
echo ""

# Run the existing setup script
bash scripts/01_setup.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Get your API keys (see API_KEYS_GUIDE.md):"
echo "   - DeepInfra:    https://deepinfra.com/dash/api_keys"
echo "   - WandB:        https://wandb.ai/authorize"
echo "   - HuggingFace:  https://huggingface.co/settings/tokens"
echo ""
echo "2. Start Jupyter:"
echo "   jupyter notebook notebooks/AdaptiveThink_Train.ipynb"
echo ""
echo "3. In the notebook:"
echo "   - Edit Cell 0: Paste your API keys"
echo "   - Set GRPO_STEPS: 200 (for pilot)"
echo "   - Click 'Run All'"
echo ""
echo "4. Read guides:"
echo "   - API_KEYS_GUIDE.md     → How to get keys"
echo "   - AZURE_SETUP_GUIDE.md  → Azure VM setup"
echo "   - SAFETY_CHECKLIST.md   → What's protected"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💰 Estimated cost for pilot run: ~\$5"
echo "⏱  Estimated time: ~12 hours"
echo ""
