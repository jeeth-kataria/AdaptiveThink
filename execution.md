# execution.md — Day-by-Day Runbook

> Operational companion to `plan.md`. Read `CLAUDE.md` first.
> 14-day window, 2 people (Jeeth = J, Ojasvi = O), 1× RTX 4090 (24 GB),
> 1× Samsung Galaxy (or Snapdragon Android fallback).
> **Tick boxes as you go.** Append surprises to §"Lessons" at the bottom.

---

## Conventions

- `[J]` / `[O]` = owner. `[J+O]` = joint review.
- Hours are wall-clock estimates; GPU hours noted separately.
- Every long run: `tmux new -s <name>`, log to `wandb`, push checkpoints to HF Hub.
- `git commit` at end of every block; PR/squash at end of stage.

---

## Day 0 — Pre-flight (do before Day 1 starts)

- [ ] [J+O] Read `StateZero_06_AdaptiveThink.pdf`, `CLAUDE.md`, `plan.md` end-to-end.
- [ ] [J] Create private GitHub repo `statezero/adaptivethink`. Push current
      `CLAUDE.md`, `plan.md`, `execution.md`, the original PDF.
- [ ] [J] Create private HF org `statezero` and three empty private repos:
      `verifier-400m`, `router-1p5b-lora`, `difficulty-labels`.
- [ ] [O] Create wandb project `statezero/adaptivethink`. Share API keys via
      a `.env.template` (committed) + `.env` (git-ignored).
- [ ] [J] Confirm GPU access (own / cloud). Note CUDA version, driver, free
      VRAM at idle. Target: CUDA 12.1+, driver 535+, ≥ 23 GB free.
- [ ] [O] Locate Galaxy device. Enable developer mode + USB debugging.
      `adb devices` should list it. If unavailable, escalate to risk #7
      fallback ladder *today*.
- [ ] [J+O] 30-min sync: align on 14-day calendar, agree daily 15-min EOD sync.

---

## Day 1 — Environment + smoke test (the day everyone underestimates)

**Goal:** by end of day, a single GRPO step runs end-to-end on a toy task.
If this slips to Day 2, the whole project slips. Treat as P0.

### Morning [J, 4 h]

- [ ] Create repo skeleton (see `CLAUDE.md` §3):
      `mkdir -p src/adaptivethink/{data,verifier,router,ttrl,eval,deploy} configs scripts tests notebooks results report`
- [ ] `python -m venv .venv && source .venv/bin/activate`
- [ ] Pin `requirements.txt`. **Last-known-good combo** (verify each on PyPI today):
      ```
      torch==2.4.1
      transformers==4.46.3
      accelerate==1.1.1
      peft==0.13.2
      bitsandbytes==0.44.1
      trl==0.12.1
      unsloth @ git+https://github.com/unslothai/unsloth.git@<pin-commit>
      vllm==0.6.4.post1
      flash-attn==2.7.0.post2  # pre-built wheel matching torch 2.4 + cu121
      datasets==3.1.0
      wandb==0.18.7
      llama-cpp-python==0.3.2
      huggingface-hub==0.26.5
      sentencepiece==0.2.0
      ```
      If any of these have moved, freeze the actually-installed versions
      and commit that. *Pin to what works, not to what looks pretty.*
- [ ] `pip install -r requirements.txt` (expect 5–15 min, expect at least
      one wheel-build failure — log it, fix it).
- [ ] Sanity:
      ```
      python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
      python -c "import unsloth, trl, vllm, peft; print(unsloth.__version__, trl.__version__, vllm.__version__, peft.__version__)"
      python -c "import flash_attn; print(flash_attn.__version__)"
      ```
- [ ] Commit `requirements.txt` + a `pip freeze > requirements.lock.txt`.

### Afternoon [J, 4 h] — GRPO smoke test

- [ ] Download the base model once and cache it:
      `huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- [ ] Write `scripts/00_smoke_grpo.py`: 50-step GRPO on 50 GSM8K items,
      reward = exact-match correctness only (no length penalty yet),
      Unsloth + TRL `GRPOTrainer` + colocated vLLM, QLoRA-4bit,
      `gpu_memory_utilization=0.45`, `group_size=4`, context 512+512.
- [ ] Run it. Watch: VRAM peak, step time, reward trend.
      **Pass criterion:** 50 steps complete without OOM, reward > 0 on at
      least one step, no crashes.
- [ ] If it fails: this is the entire job. Do not proceed to Day 2 until
      green. Most likely failures + fixes:
      - OOM → drop `group_size` to 2, drop context to 256+256.
      - vLLM colocation hang → set `enforce_eager=True`, `disable_log_stats=True`.
      - `flash-attn` import error → confirm wheel matches `torch.__version__`
        + CUDA; reinstall with `--no-build-isolation`.
      - TRL `GRPOTrainer` API mismatch → check `trl.__version__`; if API
        moved, downgrade to `trl==0.12.1` exactly or upgrade to latest and
        update the script.
- [ ] [O, evening] Mirror the env on a second machine (laptop CPU is fine,
      `pip install -r requirements.txt` only) so we have a backup author.

### EOD sync [J+O, 15 min]

- [ ] Update §"Lessons". Confirm Day 2 plan unchanged.

---

## Day 2 — Data + verifier teacher labels (kick off the long pole)

### Morning [O, 4 h] — Datasets

- [ ] `src/adaptivethink/data/loaders.py`: load GSM8K, MATH-500, StrategyQA,
      MMLU (subset). One function per dataset returning `{question, answer}`.
- [ ] Build the verifier-distillation pool: 6k GSM8K (train) + 3k MATH-500 +
      2k StrategyQA + 1k MMLU mixed = ~12k items. Save as
      `data/verifier_pool.jsonl`. Deterministic sampling with seed 0.
- [ ] Hold-out: 500 items stratified across the four sources for
      Spearman ρ eval. Save as `data/verifier_eval.jsonl`. **Never train on it.**
- [ ] Unit-test the loaders. `pytest tests/test_loaders.py`.

### Afternoon [O, 4 h] — Teacher labelling

- [ ] `scripts/02_gen_teacher_labels.py`: query teacher (DeepSeek-V3 via
      DeepInfra/Together, or GPT-4o via OpenAI) with the prompt:
      ```
      You are estimating problem difficulty for a 1.5B-parameter math/reasoning
      model (DeepSeek-R1-Distill-Qwen-1.5B). Rate this question's difficulty for
      that model on a 0.0–1.0 scale, where:
        0.0 = trivial, single-step, no chain-of-thought needed
        1.0 = needs full multi-step chain-of-thought to have any chance
      Output ONLY a JSON object: {"difficulty": <float>, "reason": "<≤20 words>"}
      Question: <Q>
      ```
- [ ] 5 calls per item, temperature 0.0–0.3, take mean of `difficulty`.
- [ ] Cache to `data/teacher_cache.sqlite` keyed on hash(prompt+model).
      Resume-safe.
- [ ] Cost guard: hard-stop if total spend > $50 (configurable).
      Print running cost every 500 items.
- [ ] Launch on a 200-item pilot first; confirm distribution looks bimodal-ish
      (we want a real spread, not all 0.5s). If teacher is mode-collapsing,
      tweak the prompt and re-pilot before launching the full run.
- [ ] Launch full 12k run in background (tmux). Expect 4–10 hours wall.
      Keep running overnight into Day 3.

### Parallel [J, 6 h] — GRPO scaffolding

- [ ] `src/adaptivethink/router/reward.py`: implement the reward function
      from `plan.md` §6 Stage-2:
      `r = correctness − λ_tok · n_tokens · (1−d) − λ_kl · KL + λ_obey · 1{decision_honoured}`
- [ ] `src/adaptivethink/router/prompt.py`: prompt template forcing the
      model to emit one of `<think>` or `<no_think>` as its first token,
      followed by reasoning (or not) and final boxed answer.
- [ ] `src/adaptivethink/router/grpo_trainer.py`: thin wrapper around TRL's
      `GRPOTrainer` injecting our reward function.
- [ ] **Unit-test the reward.** Hand-craft 6 test cases (correct+short,
      correct+long, wrong+short, wrong+long, decision-honoured, decision-
      violated). `pytest tests/test_reward.py` must pass before Day 5.
- [ ] Commit + open a draft PR for self-review.

### EOD sync

- [ ] Confirm teacher run is alive and on-budget. Confirm reward unit tests pass.

---

## Day 3 — Verifier training (Stage 1)

### Morning [O, 3 h] — Wait for teacher labels + clean

- [ ] Confirm overnight teacher run completed. Inspect distribution:
      histogram of `difficulty` should not be a delta. Target: at least 20%
      of items in each of `[0,0.33]`, `(0.33,0.67]`, `(0.67,1.0]`.
- [ ] If too narrow: re-prompt teacher on the unbalanced bins with
      stronger language ("be ruthlessly bimodal"). One extra hour, fine.
- [ ] Publish `data/verifier_pool_labelled.jsonl` to HF Hub
      `statezero/difficulty-labels` (private for now).

### Afternoon [O, 3 h GPU + 3 h wall] — Train 400M verifier

- [ ] `src/adaptivethink/verifier/model.py`: `Qwen2.5-0.5B-Instruct` encoder
      (or `microsoft/MiniLM-L12-H384` if RAM tight) + linear regression head
      on `<eos>` hidden state. Total params ~400–500M.
      *Decision criterion:* if 0.5B encoder fits and trains in <8h, use it
      (better representations); otherwise drop to MiniLM. Log decision in
      `CLAUDE.md` §5.
- [ ] `configs/verifier_distill.yaml`: AdamW, lr 2e-5, batch 32, 3 epochs,
      MSE loss + 0.2 × BCE auxiliary on threshold 0.5, weight decay 0.01,
      warmup 5%, FP16, seed 0.
- [ ] `bash scripts/03_train_verifier.sh`. Track wandb. Target: train MSE
      < 0.05, val Spearman ρ > 0.7 on the 500-item held-out.
- [ ] Acceptance gates:
      - ρ > 0.7 → ship.
      - 0.5 < ρ ≤ 0.7 → ship but flag for re-train if Stage 2 underperforms.
      - ρ ≤ 0.5 → diagnose. Most likely teacher prompt issue or label noise.
- [ ] Push `statezero/verifier-400m` (private), v0.1 tag.

### Parallel [J, 6 h] — GRPO config + pilot

- [ ] `configs/grpo_router.yaml` with the hyperparams from `plan.md` §6.
- [ ] `scripts/04_train_grpo_router.sh` calling `grpo_trainer.py`.
- [ ] **Pilot: 100 GRPO steps on 200 GSM8K items.** Look at the wandb
      curves. Reward should rise; KL should not explode; think-rate should
      not collapse to 0% or 100%.
- [ ] Common pilot pathologies + fixes:
      - `think_rate → 0` (always no-think): increase `λ_obey`, decrease `λ_tok`,
        or warm-start with 50-step SFT biased toward thinking.
      - `think_rate → 100`: increase `λ_tok`, or add explicit positive reward
        for no-think on items where verifier `d < 0.3`.
      - KL → ∞: lower lr to 5e-6, raise `λ_kl` to 1e-2.
      - Reward flat: check `correctness` extractor — boxed-answer parsing is
        the #1 silent bug.
- [ ] Once pilot looks healthy, **stop**. Do not launch the full 1500-step
      run yet — wait for the verifier to finish so we can plug in `d`.

### EOD sync

- [ ] Verifier shipped or scheduled. GRPO pilot looks healthy.
- [ ] Decision: launch full GRPO run start of Day 4 morning, or Day 4 lunch?
      Earlier is better; 36 GPU hours from launch.

---

## Day 4 — Launch full GRPO run (Stage 2 begins)

### Morning [J, 3 h] — Final pre-flight

- [ ] Plug verifier into reward: load `verifier-400m` once, run a batched
      forward pass over the prompt to compute `d` per question.
      **Cache `d` per (question, verifier_version) to disk** — recomputing
      at every GRPO step is wasted GPU.
- [ ] Sanity: run 20 GRPO steps with the *real* verifier in the loop. Make
      sure `d` actually varies across the batch (if it's all 0.5, the gating
      term is dead).
- [ ] Confirm wandb dashboards: `reward/mean`, `reward/correctness`,
      `reward/length_penalty`, `reward/obey_bonus`, `kl`, `response_length`,
      `think_rate`, plus a tiny eval `pass_at_1` every 100 steps.
- [ ] Confirm checkpointing: `save_steps=50`, `save_total_limit=3`,
      `hub_strategy=every_save`, `push_to_hub=True`. Test by killing the run
      after 60 steps and resuming — must restart cleanly from step 50.

### Late morning [J] — LAUNCH

- [ ] `tmux new -s grpo`, then `bash scripts/04_train_grpo_router.sh 2>&1 | tee logs/grpo_seed0.log`
- [ ] First seed (seed=0) only. Two more seeds run after Day 6 (we will *not*
      run 3 seeds in parallel on one GPU; we run them sequentially or on
      cloud GPUs).
- [ ] Note start time. ETA ≈ 36 h → completion mid-Day 6.

### Afternoon [O, 4 h] — Eval harness

- [ ] `src/adaptivethink/eval/benchmarks.py`: GSM8K Pass@1, Pass@k for
      k ∈ {1, 8, 64, 256}; MMLU subset (500 q stratified); StrategyQA;
      AIME24 (30 q). Identical generation config across all systems being
      compared (temperature 0.6 for Pass@1, 0.7 for Pass@k, top_p 0.95).
- [ ] `src/adaptivethink/eval/baselines.py`: implement the comparison set:
      1. `full_cot`: base R1-Distill-1.5B, always think.
      2. `no_cot`: base R1-Distill-1.5B, forced no-think.
      3. `qwen3_hybrid`: Qwen3-1.7B in hybrid mode (sanity baseline).
      4. `adaptthink_internal`: AdaptThink-style internal-confidence routing
         on R1-Distill-1.5B (re-implement the simplest version from the paper).
      5. `ours_router_only`: our GRPO router *without* the verifier feature.
      6. `ours_full`: our GRPO router *with* the verifier feature.
- [ ] Each baseline runs `eval_all` and dumps to `results/<baseline>/<bench>.json`.
- [ ] Lock the random seeds and prompts so re-runs are bit-exact for non-RL
      baselines.

### Parallel [O] — Verifier polish

- [ ] If verifier ρ ≤ 0.7 from Day 3, retrain with cleaner labels in
      background (CPU nightly). Otherwise mark v0.1 as final.
- [ ] Push verifier model card draft.

### EOD sync

- [ ] GRPO run alive, wandb green. Eval harness compiles + runs against
      `full_cot` baseline (sanity).

---

## Day 5 — Build while it cooks

GRPO run is consuming the GPU. CPU/laptop work today.

### Morning [J, 4 h] — TTRL add-on (optional, build now while waiting)

- [ ] `src/adaptivethink/ttrl/`: implement TTRL with majority-vote pseudo-
      labels per arxiv:2504.16084. N=8 samples per question, vote, treat
      majority as pseudo-label, run a single GRPO step on the unlabelled
      MMLU subset, repeat.
- [ ] **Do not run yet.** This trains *after* Stage-2 finishes. We're just
      writing the code now.
- [ ] Add a Clip-Cov / entropy-mechanism guard (arxiv:2505.22617) inside
      the TTRL loop to prevent diversity collapse.
- [ ] `configs/ttrl_addon.yaml`. Unit-test the majority-vote function.

### Afternoon [O, 4 h] — On-device pipeline (build, do not ship yet)

- [ ] `scripts/07_quantize_gguf.sh`:
      ```
      python -m llama_cpp.convert_hf_to_gguf \
        --outtype f16 --outfile work/reasoner-fp16.gguf \
        statezero/router-1p5b-merged
      ./llama-quantize work/reasoner-fp16.gguf work/reasoner-q4km.gguf Q4_K_M
      ```
      Note: requires merging the LoRA adapter into the base before convert.
- [ ] Run on the *base* (un-RL-trained) model end-to-end as a dry run.
      Confirm we can quantise + load + generate via `llama-cli`. Measure
      host-side latency on CPU + 4090 for sanity.
- [ ] `scripts/08_galaxy_bench.sh`: ADB push of GGUFs + a 200-question
      battery script. Each question logs:
      `t_first_token_ms`, `t_total_ms`, `peak_rss_mb`, `n_input_tokens`,
      `n_output_tokens`, `decision`, `correct`. Output → `results/onDevice/`.
- [ ] Test the ADB harness against the *base* model on the actual phone
      (or Snapdragon fallback) — yes, today, while the GPU is busy. We need
      to discover device-side issues now, not on Day 11.

### Parallel [J, 2 h] — Watch GRPO

- [ ] Eyeball wandb every few hours. If something's clearly broken (KL
      explosion, reward flat for 200 steps, think_rate frozen), kill, fix,
      relaunch. *Don't* let a known-broken run finish — that's 30 wasted h.
- [ ] If reward looks plateaued but everything else healthy, let it run.
      RL plateaus and then sometimes climbs again.

### EOD sync

- [ ] TTRL code complete + tested. On-device pipeline tested with base model.
      GRPO at ~step 700 / 1500.

---

## Day 6 — GRPO finishes; first real numbers

### Morning [J, 4 h] — Land the seed-0 run

- [ ] By ~10:00, GRPO seed-0 should be near completion. Let it finish to step
      1500 unless reward has been falling for 300 steps (then revert to best
      checkpoint).
- [ ] Pick "best" checkpoint by **eval Pass@1 + length** Pareto on a held-out
      set, *not* by raw reward (reward can be hacked).
- [ ] Merge LoRA → push `statezero/router-1p5b-merged` v0.1.
- [ ] Run **full eval suite** on `ours_full` (seed-0 only, single point).
- [ ] First Pareto sketch with the 6 baselines. *This is the moment of truth.*

### Decision tree based on first Pareto

- **Case A — `ours_full` strictly dominates `full_cot` on Pareto:** good. Lock
  hyperparams. Launch seeds 1 + 2 in series (Days 6 evening + Day 7 evening).
- **Case B — `ours_full` close but not dominating:** sweep `λ_tok` ∈
  {2e-4, 1e-3} as two short 600-step runs (~12 h each). Pick best. Then
  seeds 1+2.
- **Case C — `ours_full` no better than `full_cot` at any point:** root-cause
  fast. Most likely culprits: (i) verifier `d` not informative — check
  Spearman against actual model accuracy; (ii) reward parsing — recompute on
  100 hand-graded items; (iii) think_rate stuck. Fix and relaunch a 600-step
  run. Day 7 will be tight but recoverable.
- **Case D — total disaster:** fall back to AdaptThink-style internal-
  confidence routing without the verifier (we lose one of three deltas but
  still ship a working system). Document honestly in the report.

### Afternoon [O, 4 h] — Run TTRL add-on on seed-0 checkpoint

- [ ] `bash scripts/05_ttrl_addon.sh` against `ours_full` on the MMLU
      unlabelled subset. ~6–8 h on the GPU.
- [ ] Add `ours_full_ttrl` to the eval matrix.

### Parallel [J] — Launch seed-1

- [ ] If Case A or B, queue seed-1 to start once TTRL releases the GPU
      (evening / overnight).

### EOD sync

- [ ] Confirm we have at least one full row of Pareto numbers. Decide A/B/C/D.

---

## Day 7 — Multi-seed + ablations

### Morning [J, 4 h] — Seed-1 finishing, seed-2 launching

- [ ] Land seed-1; eval; push.
- [ ] Launch seed-2 (will run into Day 8 morning).
- [ ] If we are in Case B and the sweep helped, the "winning" λ becomes the
      pinned value — remember to update `configs/grpo_router.yaml`.

### Afternoon [O, 4 h] — Ablations on seed-0

These are short runs (300–500 steps each) used as ablations, not as
headline numbers. The headline is 3-seed `ours_full`.

- [ ] **Ablation 1:** turn off length penalty (`λ_tok = 0`). Expect Pareto
      collapse to `full_cot`.
- [ ] **Ablation 2:** turn off verifier gating (drop `(1−d)` term, keep
      `λ_tok · n_tokens`). Expect performance to degrade on hard items.
- [ ] **Ablation 3:** turn off obey bonus. Expect think_rate to decouple from
      decision token, validating the bonus.
- [ ] **Ablation 4:** routing input = prompt only (no verifier feature).
      Compare to `ours_full`. This is the "AdaptThink-style" comparison.
- [ ] All ablations run sequentially overnight. Each ~3 h GPU.

### Parallel — Report writing begins (yes, now)

- [ ] [O, 2 h] Draft `report/sections/01_intro.md`,
      `02_related_work.md` (cite every paper from `plan.md` §2.1 + §2.2),
      `03_method.md` skeleton with reward formula and architecture diagram.

### EOD sync

- [ ] Two seeds done. Ablations queued. Report skeleton up.

---

## Day 8 — Lock results; on-device begins for real

### Morning [J, 4 h] — Seed-2 lands

- [ ] Land seed-2; eval; push.
- [ ] We now have 3-seed numbers for `ours_full`. Compute mean ± 95% CI for
      every benchmark. **This is the table that goes in the report.**
- [ ] Lock the `ours_full` checkpoint version. Tag `v1.0` on HF Hub.
      Anything beyond this is a *better* version, not a replacement.
- [ ] Lock the Pareto chart's six points. (`ours_full_ttrl` is added later
      as a seventh point if it helps.)

### Afternoon [J, 4 h] — On-device push

- [ ] Re-run the merge + GGUF quantise pipeline on the locked `v1.0`
      checkpoint. Verify host-side parity (`llama.cpp` host eval vs. HF
      generation) on 50 GSM8K items: accuracy delta < 1.5% is acceptable;
      > 3% means a quant/merge bug.
- [ ] Push `reasoner-q4km.gguf` (≈ 1.0–1.2 GB) and `verifier-fp16.gguf`
      (≈ 800 MB) to phone via ADB.
- [ ] Run the 200-question on-device battery (GSM8K stratified by verifier
      `d`). Log everything from `scripts/08_galaxy_bench.sh`.
- [ ] First mobile latency / accuracy point on the Pareto chart. **This is
      the chart that wins us the comp** — make sure the harness is correct.

### Parallel [O, 6 h] — Ablation results come in; `ours_full_ttrl` lands

- [ ] Compile ablations 1–4 into `results/ablations/table.csv`.
- [ ] If TTRL helps on any benchmark, add `ours_full_ttrl` row + Pareto point.
      If it doesn't help, mark explicitly in the report — negative results
      are honest.

### EOD sync

- [ ] 3-seed locked. On-device first pass done. Pareto has 6–7 points.

---

## Day 9 — Mobile-side polish + battery / RAM benchmarks

### Morning [J, 4 h] — Battery + thermal

- [ ] On the Galaxy: charge to 100%, unplug, run 100-question GSM8K with
      `full_cot` only. Log `dumpsys batterystats` before/after. Repeat with
      `ours_full`. Ratio = our battery win factor. Expect 2–3× depending on
      `think_rate`.
- [ ] Watch thermal throttling: use `dumpsys thermalservice` mid-run. If
      throttled, our latency numbers are slow-clock numbers — note this.
- [ ] Peak RAM during inference: `cat /proc/<pid>/status | grep VmPeak`.
- [ ] Consolidate into `results/onDevice/galaxy_pareto.csv` and a single
      Pareto + bar chart (`results/pareto/galaxy_pareto.png`) using
      `matplotlib`. One chart, one decision: which point dominates and where.

### Afternoon [O, 4 h] — Demo video shoot (rough cut)

- [ ] Storyboard (target 3:00):
      0:00–0:20 — problem ("SLMs over-think; same compute every query")
      0:20–0:50 — system diagram (verifier → router → reasoner)
      0:50–1:30 — Pareto chart explained
      1:30–2:30 — **live phone demo**: easy q (no-think, fast) vs. hard q
      (think, slower, correct). Side-by-side with full-CoT for contrast.
      2:30–3:00 — repo / artefacts / call to reproduce
- [ ] Shoot the phone segment with a tripod, screen-record the Galaxy
      directly (`adb shell screenrecord`) for clean output, not handheld.
- [ ] Voice-over by J (clear), captions by O.
- [ ] First cut today; final cut Day 13.

### Parallel [J] — Report writing

- [ ] Sections `04_experiments.md`, `05_results.md` with the locked tables.
- [ ] Reserve the headline number slot in the title. Pick the number from
      the Pareto: e.g. *"AdaptiveThink: 47% token reduction at −0.3% accuracy
      on GSM8K, on a Samsung Galaxy"* (replace 47 / 0.3 with real numbers).

### EOD sync

- [ ] Mobile Pareto locked. Rough demo cut exists. Report half-drafted.

---

## Day 10 — Nice-to-haves + buffer

This day is deliberately under-scheduled. **Resist the urge to start a new
training run.** Code freeze happens tomorrow.

### Pick from this menu (in order; stop when out of time)

- [ ] [J, 3 h] One more sweep point if Pareto is close: try
      `λ_tok = 7e-4` for 600 steps as a tiebreaker. Only if it would clearly
      improve the headline number.
- [ ] [O, 3 h] Battery curve at *intermediate* `λ_tok` values to show the
      knob works (a sweep figure for the report).
- [ ] [J+O, 3 h] Cross-architecture sanity: run `ours_full` against
      `Qwen3-1.7B` (already supports hybrid thinking) on GSM8K. Even a
      modest improvement here is a strong demo of generalisation, but only
      if seed-0 alone gives a clean signal — no time for 3 seeds.
- [ ] [O, 2 h] StrategyQA + AIME24 on-device numbers if the phone holds up.
- [ ] [J, 2 h] Failure-case gallery: 5 hand-picked qualitative examples
      (good think, good no-think, bad think, bad no-think, edge case). Goes
      in report appendix.

### Buffer reserved for things that broke

History says 1–2 of these will be needed:
- Re-eval on a benchmark whose extractor was buggy.
- Re-quantise after fixing a tokenizer mismatch in conversion.
- Re-shoot a demo segment.
- A teammate's machine dies and we need to mirror state.

### EOD sync

- [ ] Confirm everything for the headline result is **on disk + on HF Hub +
      committed to git**. Tomorrow is freeze day.

---

## Day 11 — Code freeze. Ablation + reproducibility audit.

### Morning [J+O, 4 h] — Freeze

- [ ] **No new training runs after noon today.** No new sweeps. No new
      benchmarks. If a number isn't in `results/` by lunch, it isn't in the
      report.
- [ ] Tag the repo `v1.0-frozen`. Tag every HF artefact `v1.0`.
- [ ] `pip freeze > requirements.lock.txt` from the actual training box.
      Commit it.

### Afternoon [J, 4 h] — Reproducibility audit

- [ ] On a **fresh clone** + fresh venv (use the laptop or a cloud box):
      ```
      git clone <repo>; cd adaptivethink
      python -m venv .venv && source .venv/bin/activate
      pip install -r requirements.lock.txt
      bash scripts/01_setup.sh
      bash scripts/06_eval_all.sh --quick   # subsample for speed
      ```
      Confirm headline numbers reproduce within CI. If they don't, fix it
      *today*. This is what the judges will hit.
- [ ] Write `README.md` for the repo:
      - 60-second pitch
      - Architecture diagram (one image)
      - Headline number badge
      - Quickstart (`bash scripts/06_eval_all.sh --quick`)
      - Pointers to HF artefacts
      - Citations
      - Honest "Limitations" section
      - Apache-2.0 + `CITATION.cff` link

### Parallel [O, 4 h] — Ablation table, finalised

- [ ] Final ablation table with mean ± 95% CI over 3 seeds for the headline
      result; single seed for ablations is acceptable but mark it.
- [ ] Final Pareto chart with the 7 points (full_cot, no_cot, qwen3_hybrid,
      adaptthink_internal, ours_router_only, ours_full, ours_full_ttrl).
- [ ] Save vector versions (`.pdf`, `.svg`) for the report; PNG for the
      README and video.

### EOD sync

- [ ] Reproducibility audit green. Tables + charts final.

---

## Day 12 — Technical report

### All day [J+O, 8 h, split]

Target: **8 pages max**, NeurIPS / arXiv-style template (single column also
fine). One PDF, in `report/AdaptiveThink_report.pdf`, source in `report/tex/`.

Section budget:
- Title + abstract — 0.25 page. **Headline number in title.**
- §1 Introduction — 0.75 page. The *actual* problem (over-thinking on easy q).
- §2 Related Work — 1 page. Honest. AdaptThink, Qwen3, R2R, Weaver, TTRL,
  Adaptive Length Penalties, TAB. Cite by arxiv ID.
- §3 Method — 2 pages. Architecture diagram, reward formula, training recipe,
  on-device pipeline.
- §4 Experiments — 1 page. Datasets, baselines, metrics, hardware.
- §5 Results — 2 pages. Pareto chart (full-page-width), main table, ablation
  table, on-device latency / RAM / battery chart.
- §6 Discussion + Limitations — 0.5 page. **Be explicit** about what we
  didn't beat (e.g. AdaptThink in pure accuracy at large compute) and where
  we win (system-level, on-device).
- §7 Conclusion + reproducibility — 0.5 page. HF Hub links, repo link,
  Apache-2.0, license, contact.
- References.

### Hard rules for the report

- [ ] **Every number in the body is in a CSV in the repo.** No exceptions.
      Reviewers will check.
- [ ] **No claim "first to RL-train a router".** See `CLAUDE.md` §2 + `plan.md` §2.1.
- [ ] **Every prior-work paper has an arxiv ID.** No "personal communication".
- [ ] **One headline number, not three.** Pick the most defensible
      (probably GSM8K token reduction at fixed accuracy on Galaxy).
- [ ] **Compile and proofread on a fresh machine.** Latex bugs found by judges
      are a sign of a rushed submission.

### EOD

- [ ] Report v1 PDF compiled. Send to a 3rd party (a friend / mentor) for
      a 30-minute red-team read overnight.

---

## Day 13 — Video + repo polish

### Morning [O, 4 h] — Video final cut

- [ ] Incorporate red-team feedback on the report (parallel with J).
- [ ] Re-shoot any phone segments that look cluttered.
- [ ] Add captions / accessibility text.
- [ ] Render at 1080p, ≤ 100 MB target so it uploads cleanly.
- [ ] Upload to **YouTube unlisted** (NOT Drive — see `CLAUDE.md` §4.2).
      Record the URL.
- [ ] Add the URL to README, report, and submission form.

### Morning [J, 4 h] — Report v2

- [ ] Address all red-team comments. Tighten. Cut anything that doesn't
      serve the headline number.
- [ ] Re-compile. Spell-check. Page count check.
- [ ] Final author affiliations + emails. Final contact in §7.

### Afternoon [J+O, 4 h] — Repo polish

- [ ] README final pass with embedded video, embedded Pareto chart, badges.
- [ ] HF model cards: every artefact (`verifier-400m`, `router-1p5b-lora`,
      `router-1p5b-merged`, `difficulty-labels` dataset) has a card with:
      training recipe, eval numbers, intended use, limitations, license,
      citation. Reviewers will spot-check at least one.
- [ ] `CITATION.cff` filled in (preferred-citation = the report).
- [ ] `LICENSE` (Apache-2.0) + `NOTICE` (third-party attributions).
- [ ] `.gitignore` final pass — make sure no `wandb/` runs, no `outputs/`
      checkpoints, no `.env`, no API keys.
- [ ] One last `git status` audit + `git log` cleanup (squash trivial
      "wip" commits).

### EOD

- [ ] Video uploaded. Report v2 PDF in repo. Repo polished. Tag `v1.0-submission`.

---

## Day 14 — Submit + buffer

### Morning [J+O, 3 h] — Submission

- [ ] Re-read the EnnovateX Round-2 submission form *carefully*. Note word
      limits, file-size limits, required sections.
- [ ] Submit:
      - GitHub repo URL
      - HF Hub artefact URLs (4)
      - YouTube demo URL (unlisted is fine; **not Drive**)
      - Report PDF (8 pages)
      - Pareto chart as separate image (some forms ask for it)
      - Team details (already in the deck)
- [ ] Confirm submission received (screenshot the confirmation page).

### Afternoon — Buffer

- [ ] Anything that broke in the last 24 h.
- [ ] Tweet / LinkedIn announcement if the rules permit (some hackathons
      enforce embargo until the result; check first).
- [ ] Thank-you note to anyone who red-teamed for us.

### EOD

- [ ] Done. Sleep.

---

## Appendix A — Command cheatsheet (verbatim, copy-pasteable)

```bash
# Day 1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, unsloth, trl, vllm; print('ok')"
python scripts/00_smoke_grpo.py

# Day 2
python scripts/02_gen_teacher_labels.py --config configs/verifier_distill.yaml \
  --pool data/verifier_pool.jsonl --out data/teacher_labels.jsonl \
  --max-cost-usd 50

# Day 3
bash scripts/03_train_verifier.sh
python -m adaptivethink.eval.verifier_eval \
  --ckpt outputs/verifier-400m/best --eval data/verifier_eval.jsonl

# Day 4
tmux new -s grpo
bash scripts/04_train_grpo_router.sh seed=0 2>&1 | tee logs/grpo_seed0.log
# Ctrl-b d to detach; tmux attach -t grpo to reattach

# Days 6–7 — additional seeds
bash scripts/04_train_grpo_router.sh seed=1
bash scripts/04_train_grpo_router.sh seed=2

# TTRL add-on
bash scripts/05_ttrl_addon.sh --base statezero/router-1p5b-merged

# Eval everything
bash scripts/06_eval_all.sh

# Quantise
bash scripts/07_quantize_gguf.sh

# On-device benchmark
adb devices  # confirm phone connected
bash scripts/08_galaxy_bench.sh
```

---

## Appendix B — Troubleshooting

**`flash-attn` ImportError after install.** Wheel doesn't match torch ABI.
Reinstall: `pip uninstall flash-attn -y && pip install flash-attn==2.7.0.post2 --no-build-isolation --no-cache-dir`. Confirm `torch.version.cuda` matches the wheel's CUDA tag.

**vLLM colocation hangs at first rollout.** Set `enforce_eager=True`,
`disable_log_stats=True`, `gpu_memory_utilization=0.40` (not 0.45). If still
hanging, swap to non-colocated vLLM on a second small GPU, or fall back to
HF generation for rollouts (slower but works).

**OOM at GRPO step.** In order of preference: (1) drop context to 768+768,
(2) drop `group_size` from 8 → 4, (3) enable `gradient_checkpointing=True`,
(4) drop `gpu_memory_utilization` and accept slower rollouts.

**GRPO reward stuck at 0.** Boxed-answer extractor bug. Print 20 random
generations + extracted answers + ground-truth side by side. The bug is
almost always there.

**`think_rate` collapses to 0% or 100%.** See `plan.md` §6 and Day 3 fixes.
Also try warm-starting from a 50-step SFT pass that biases toward the
opposite mode of whatever it collapsed to.

**Verifier ρ too low.** Teacher prompt is too vague. Make it bimodal:
"output 0.05 if a 7-year-old could solve it, 0.95 if it requires explicit
multi-step CoT". 5 calls per item, take median not mean.

**GGUF host eval ≠ HF eval.** Tokenizer mismatch (chat template lost during
merge), or the LoRA wasn't actually merged. Re-do
`peft.PeftModel.from_pretrained(...).merge_and_unload()` then save.

**ADB push fails / Galaxy out of storage.** Quantise verifier to Q5_K_M
(saves ~400 MB) or skip pushing the FP32 reasoner backup. We need only
the Q4_K_M reasoner + the verifier.

**Phone hits thermal throttle mid-benchmark.** Pause 60 s every 20
questions; document this in the report. Numbers from a throttled phone are
*lower-bound* on what shipped silicon would do.

---

## Appendix C — Lessons (append as you learn)

> Date — what surprised us — what we did about it.

- *(empty until execution begins)*

---

## Appendix D — One-page schedule overview

```
Day  Owner Focus                                    GPU?  Deliverable
---- ----- --------------------------------------   ----  --------------------------
0    J+O   Pre-flight                               no    repo, hf, wandb, phone
1    J     Env + GRPO smoke test                    yes   50-step run green
2    O+J   Data + teacher labels + GRPO scaffold    yes   12k labels in flight
3    O+J   Verifier train + GRPO pilot              yes   verifier ρ>0.7, pilot OK
4    J+O   Launch full GRPO seed-0 + eval harness   yes   seed-0 running, baselines ready
5    J+O   TTRL code + on-device dry-run            yes   GPU busy; mobile ready
6    J+O   Land seed-0; first Pareto                yes   Pareto v0; A/B/C/D decision
7    J+O   Seeds 1+2 + ablations + report draft     yes   3-seed numbers
8    J+O   Lock results + on-device push            yes   mobile Pareto first cut
9    J+O   Battery/RAM + demo rough cut             phone galaxy pareto + video v0
10   J+O   Buffer / nice-to-haves                   maybe stable system
11   J+O   Code freeze + repro audit                no    fresh-clone audit green
12   J+O   Report v1                                no    8-page PDF
13   J+O   Video final + repo polish                no    YouTube + clean repo
14   J+O   Submit + buffer                          no    submission confirmed
```
