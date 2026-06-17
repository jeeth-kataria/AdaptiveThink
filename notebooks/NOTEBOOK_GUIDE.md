# AdaptiveThink Training Notebook — Usage Guide

## ✅ Phase 1-4 Complete

All blocking issues from Phase 1 audit have been fixed and **Phase 4 is complete**: the notebook `notebooks/AdaptiveThink_Train.ipynb` is written and ready.

---

## Notebook Structure

**36 cells total:**
- **11 markdown section headers**
- **25 executable code cells**

### Sections

0. **Config (Cell 0)** — the ONLY cell users edit
1. **Setup (Cells 1-5)** — env audit, NVMe mount, secrets, deps, unit tests
2. **Data (Cells 6-8)** — downloads, teacher labels, GRPO data build
3. **Verifier (Cells 9-10)** — 400M difficulty verifier training + HF push
4. **GRPO Router (Cells 11-16)** — smoke test, 3 seeds, Pareto check, CI aggregation
5. **TTRL (Cell 17)** — optional test-time RL
6. **Eval (Cells 18-19)** — full benchmark suite + Pareto charts
7. **Quantise (Cell 20)** — GGUF Q4_K_M export + parity check
8. **On-device (Cell 21)** — stub for Day 8+ Galaxy benchmarks
9. **Upload + Wrap-up (Cells 22-24)** — HF push, progress docs, VM deallocate

---

## Key Features

### Robustness

✅ **Auto-resume from checkpoints** — every training cell saves progress every 50 steps  
✅ **Exception handling** — try/except with wandb logging + resume commands  
✅ **Hard gates** — unit tests (Cell 5), smoke test (Cell 11), verifier ρ check (Cell 9)  
✅ **Parity verification** — GGUF vs HF accuracy check before accepting quantised model  
✅ **Case A/B/C/D decision tree** — Cell 13 diagnoses seed-0 results and guides next steps  

### Error-free design

✅ **TRL 0.12.1 API fix** — `vllm_server_kwargs` not direct kwarg (fixed in Cells 11-12)  
✅ **Data pipeline** — auto-builds `gsm8k_train_labelled.jsonl` if missing  
✅ **Distribution checks** — warns if difficulty labels mode-collapsed to 0.5  
✅ **VRAM auto-config** — adapts group_size/seq_len/vllm based on detected GPU  

### Azure-optimised

✅ **NVMe auto-mount** — detects + mounts `/dev/nvme0n1` for fast I/O  
✅ **Azure Key Vault support** — Managed Identity secret injection (no credentials in code)  
✅ **Spot eviction resilience** — HF Hub checkpoint backup every 50 steps  
✅ **Auto-deallocate** — optional VM shutdown at end to stop billing  

---

## How to Use

### On Azure NC24ads_A100_v4

1. **Spin up VM** (spot, 80 GB A100, ~$0.75/hr):
   ```bash
   az vm create -g <rg> -n adaptivethink-vm \
     --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
     --size Standard_NC24ads_A100_v4 \
     --priority Spot --max-price 1.0 \
     --assign-identity --generate-ssh-keys
   ```

2. **SSH into VM**:
   ```bash
   ssh azureuser@<vm-ip>
   ```

3. **Clone repo**:
   ```bash
   git clone https://github.com/jeeth-kataria/AdaptiveThink.git
   cd AdaptiveThink
   ```

4. **Start Jupyter**:
   ```bash
   jupyter notebook --no-browser --port=8888
   # Then SSH tunnel from your laptop:
   # ssh -L 8888:localhost:8888 azureuser@<vm-ip>
   ```

5. **Open `notebooks/AdaptiveThink_Train.ipynb`**

6. **Edit Cell 0 ONLY:**
   - Fill in API keys (or set `AZURE_KV_URL` for Key Vault)
   - Set `AUTO_DEALLOCATE=True` + Azure VM details to auto-shutdown at end
   - Review hyperparams (defaults are good)

7. **Run all cells** (Kernel → Restart & Run All)

8. **Walk away.** Total time: ~120 GPU hours (~5 days wall time for 3 seeds sequential).

---

## Estimated Costs (Azure Spot)

| Stage | Hours | Rate | Cost |
|---|---|---|---|
| Teacher labels (D4s_v5) | 6h | $0.05 | $0.30 |
| Verifier (NC4as_T4_v3) | 8h | $0.20 | $1.60 |
| GRPO 3 seeds (NC24ads_A100_v4) | 108h | $0.75 | $81.00 |
| Eval + quant (NC4as_T4_v3) | 8h | $0.20 | $1.60 |
| **Total** | | | **~$85** |

Add 50% buffer for spot evictions → budget $130.

---

## Outputs

After completion, you will have:

- ✅ `results/figures/kpi_table.md` — Pass/Fail on ≥+5% on ≥2 benchmarks
- ✅ `results/figures/pareto_gsm8k.png` — accuracy vs compute chart
- ✅ `results/ablations/3seed_ci_table.csv` — mean ± 95% CI per benchmark
- ✅ `outputs/gguf/router-1p5b-Q4_K_M.gguf` — quantised model (~1.2 GB)
- ✅ All checkpoints pushed to HF Hub under `statezero/*`
- ✅ `results/progress.md`, `CLAUDE.md`, `execution.md` auto-updated

---

## What if something fails?

Every training cell has the same pattern:

```python
try:
    subprocess.run([...], check=True)
except Exception as e:
    print(f"❌ Failed: {e}")
    print(f"RESUME COMMAND: <exact command to rerun>")
    raise
```

**To resume:**
1. Read the "RESUME COMMAND" printed in the output
2. Either re-run the same cell (it will auto-detect checkpoints), or
3. Run the command in a terminal inside a `tmux` session

**Common failures + fixes are diagnosed inline** — e.g. OOM → reduce group_size, verifier ρ low → check teacher labels, etc.

---

## Self-Review Notes

**Cells I'm most confident about:**
- Cells 0-8 (setup + data) — straightforward, no complex logic
- Cell 9 (verifier train) — mirrors existing `train.py`, just wrapped
- Cells 11, 20 (smoke test, GGUF) — directly call existing scripts

**Cells that need field-testing:**
- Cell 12-13 (GRPO runner + Pareto check) — subprocess streaming + Case logic
- Cell 16 (3-seed CI) — statistics.mean/stdev on small n=3
- Cell 19 (Pareto chart) — IPython.display may not render in some Jupyter setups

**Known limitations:**
- Cell 21 (on-device) is a stub — scripts/08_galaxy_bench.sh doesn't exist yet
- No Dockerfile cell (could add as Cell 0.5 for reproducibility)
- wandb init happens inside train_grpo.py, not in the notebook (could surface it)

---

## Next Steps

1. ✅ **Phase 1-4 complete** — notebook is written
2. ⏭️ **Field test** — run Cells 0-10 on a T4 VM to verify Stages 0-1 work
3. ⏭️ **Full run** — launch on A100 spot VM for the 36h seed-0 run
4. 📝 **Report writing** (Day 12) — results/ will have all numbers needed

Ready for execution. 🚀
