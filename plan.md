# plan.md — StateZero / AdaptiveThink, Round 2 Strategy

> **Goal:** win Samsung EnnovateX 2025 PS06 (RL for SLM Reasoning) Round 2.
> **Team:** StateZero — Jeeth Bhavesh Kataria, Ojasvi Poonia, Ramaiah IT, Bangalore.
> **Window:** 14 days (per Round-1 timeline), single 24 GB GPU (RTX 4090) +
> 1× Samsung Galaxy device for on-device benchmarks.
> **Read this with:** `CLAUDE.md` (rules), `execution.md` (day-by-day runbook).

---

## 1. One-paragraph pitch (the version we send the judges)

> SLMs (1.5B–7B) trained with RL waste compute by reasoning at full chain-of-
> thought length on every query — easy or hard. **AdaptiveThink** is an
> end-to-end on-device system that pairs a `DeepSeek-R1-Distill-Qwen-1.5B`
> reasoner with a 400M Weaver-style distilled difficulty verifier and an
> RL-trained 2-token routing head, the whole thing quantised to GGUF Q4_K_M
> and shipped on a Samsung Galaxy. The router learns *when* to think, the
> verifier provides a calibrated external signal, and the phone runs it
> offline. We deliver a measured latency / accuracy Pareto curve on consumer
> mobile hardware — the first published end-to-end on-device adaptive-reasoning
> system with an external distilled verifier.

---

## 2. What's actually novel (and what isn't)

### 2.1 Not novel anymore — be honest

| Claim from Round-1 deck | Status | Killer paper |
|---|---|---|
| "First to RL-train a routing model" | **Dead** | AdaptThink, arxiv:2505.13417 (May 2025) |
| "RL with `correctness − λ·tokens` reward" | **Dead** | Efficient Reasoning w/ Adaptive Length Penalties, arxiv:2506.05256 |
| "Difficulty-gated length penalty" | **Dead** | CODA, arxiv:2603.08659 (Mar 2026) — uses difficulty gates on length penalty |
| "Adaptive thinking on SLMs" | **Dead** | Qwen3 hybrid thinking, arxiv:2505.09388 |
| "Tiny verifier" | **Done by Stanford** | Weaver, arxiv:2506.18203 (Jun 2025) |
| "TTRL on test data" | **Done** | TTRL, NeurIPS 2025, arxiv:2504.16084 |

**Critical distinction that keeps our novelty alive:**
CODA's difficulty signal is **policy-internal** (estimated from group-rollout pass-rate).
AdaptThink's difficulty signal is **internal model confidence**.
**Neither uses an external, separately-trained distilled verifier as the difficulty signal.**
Our `(1−d)` gating term uses `d` from the Weaver-style 400M cross-encoder — a signal
that is calibrated independently of the policy being trained, making it more stable
under distribution shift and quantisation.

If we had submitted the original framing in Round 2 it would have been
trivially shot down. The pivot below is mandatory, not optional.

### 2.2 What we actually contribute (defensible novelty)

> **System contribution:** the first reproducible, fully on-device adaptive-
> reasoning pipeline that **composes** (a) a separately distilled 400M external
> Weaver-style difficulty verifier, (b) an RL-trained 2-token routing head on a
> 1.5B reasoner using GRPO with a length-penalised reward, (c) optional TTRL
> test-time scaling, and (d) GGUF Q4_K_M deployment on a Samsung Galaxy with
> measured latency / RAM / battery numbers.
>
> No prior work ships all four pieces together with an empirical mobile Pareto.

We additionally contribute three smaller-but-real research artefacts:

1. **Verifier-conditioned reward shaping.** AdaptThink's router uses internal
   model confidence to decide when to think. We feed the *external* verifier's
   calibrated difficulty score `d ∈ [0, 1]` directly into the GRPO reward as a
   gating term: `r = correctness − λ · tokens · (1 − d)`. Easy questions
   (`d ≈ 0`) pay a heavy length penalty; hard questions (`d ≈ 1`) effectively
   pay none, decoupling routing from confidence collapse. To our knowledge no
   public paper does this — it's a simple, defensible delta over AdaptThink.

2. **Cross-encoder-as-router-feature ablation.** We compare three router input
   regimes head-to-head: (i) prompt-only (AdaptThink-style), (ii) prompt +
   internal logits (Qwen3-hybrid-style), (iii) prompt + external Weaver
   verifier score (ours). We expect (iii) to dominate the Pareto, especially
   under quantisation where internal-logit signals get noisier.

3. **Mobile Pareto chart.** A single chart, run on real Galaxy hardware, with
   five points: full-CoT, no-CoT, Qwen3-hybrid mode, AdaptThink-style internal
   routing, and our verifier-routed system. Plus a battery-drain bar. This is
   what the judges remember.

### 2.3 The honest framing for the report

> "Recent work (AdaptThink, Qwen3, R2R, TAB, Adaptive Length Penalties) has
> independently demonstrated that RL-trained adaptive reasoning is feasible
> for SLMs. We do not contest that prior art. Our contribution is to *integrate*
> these ideas into a reproducible, fully on-device pipeline, and to ship and
> measure it on a Samsung Galaxy — a step the literature has not yet taken."

Judges (and reviewers) reward honesty about prior art far more than they reward
inflated novelty claims that fall apart in 30 seconds of search.

---

## 3. Strategic advantages we can press

1. **Samsung Galaxy mapping is a freebie we'd be insane not to use.**
   - Samsung SAIL Montréal published TRM (7M-param recursive reasoning) on
     Oct 2025 — Samsung Research itself is investing heavily in tiny on-device
     reasoning.
   - Samsung's PIM-SHERPA (Nov 2025) targets on-device LLM inference via
     Processing-in-Memory.
   - Last year's EnnovateX 2025 winner (Team COD3INE) shipped an on-device
     multi-agent product mapped to Galaxy.
   - **Implication:** an on-device demo on a Galaxy phone is the strongest
     possible Samsung-judging-criterion alignment. Spend the engineering hours.

2. **Round-1 deck already named the right tools.** Unsloth, TRL GRPOTrainer,
   colocated vLLM, QLoRA-4bit, FlashAttention-2, GGUF Q4_K_M. These compose
   on a single 24 GB GPU. We do not need to invent a new stack.

3. **DeepSeek-R1-Distill-Qwen-1.5B is the correct base.** AIME24 28.9 Pass@1,
   MATH-500 83.9, and Qwen2.5-Math is exceptionally responsive to RL per
   SimpleRL-Zoo and 1-shot RLVR. Don't second-guess this choice.

4. **Reproducibility = points.** Most hackathon submissions are hacky one-shot
   colab notebooks. A pinned `requirements.txt`, 3-seed runs with confidence
   intervals, public HF model card + dataset card, a one-command repro
   (`bash scripts/06_eval_all.sh`), and a `CITATION.cff` will dominate the
   "engineering quality" axis.

---

## 4. Judging-criteria mapping (assumed; refine if Round-2 brief differs)

EnnovateX-style hackathons typically score on five axes. Our coverage:

| Axis | Our move | Evidence in deliverable |
|---|---|---|
| **Novelty / innovation** | System composition + verifier-conditioned reward + on-device Pareto | §2.2 above; report §4–5 |
| **Technical depth / RL rigour** | GRPO with TRL 0.12+, QLoRA-4bit, colocated vLLM, 3-seed CIs, KL-β sweep, reward-component ablation | report §6, ablation table, training curves |
| **Samsung product alignment** | Galaxy on-device math/coding tutor; offline; private; battery-aware | demo video on Galaxy; latency/RAM/battery chart |
| **Reproducibility / openness** | Apache-2.0; pinned reqs; HF model + dataset cards; `train.sh` + `eval.sh`; `CITATION.cff` | repo root |
| **Communication** | 8-page report w/ headline number in title; 3-min YouTube demo; clean README; one-screen architecture diagram | `report/`, video, README |

If the Round-2 brief publishes explicit weights, re-rank the day plan in
`execution.md` accordingly.

---

## 5. Risk register (with mitigations)

Ordered by `probability × impact`.

| # | Risk | P | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **GRPO run diverges or reward-hacks** (always answers no-think) | High | Project-killing | Pilot run @ 100 steps before launching 1500-step run; clip reward; warm-start from a 50-step SFT to bias toward thinking; KL-β sweep (0.001 / 0.005 / 0.02); reward = correctness − λ·tokens·(1−d) prevents collapse on hard items |
| 2 | **Stack incompatibility** (Unsloth × TRL × vLLM version skew) | High | 1–2 day delay | Pin exact versions on day 1; smoke-test full GRPO step on day 1, not day 5; keep a known-good `pip freeze` snapshot |
| 3 | **24 GB VRAM OOM** at GRPO rollouts | Med | 6 h delay each occurrence | Unsloth + QLoRA-4bit (≈ 5–7 GB); short context (1024 prompt + 1024 gen); group_size = 4 → 8; gradient checkpointing on; vLLM `gpu_memory_utilization=0.45` colocated |
| 4 | **Teacher-label budget blow-up** (DeepSeek-V3 / GPT-4o API for 12k items) | Med | $50–200 surprise | Cap to 8k labelled items for verifier; use temperature 0; cache aggressively; batched calls; if budget tight, swap GPT-4o for `Qwen2.5-72B-Instruct` open weights via Together AI / DeepInfra |
| 5 | **AdaptThink does our exact thing better** | Med | Novelty deflation | We do not compete on the routing algorithm; we compete on the *system + on-device Pareto*. Cite AdaptThink positively in §2.2; add it as a baseline in our Pareto chart |
| 6 | **GGUF Q4_K_M kills the verifier accuracy** (cross-encoder is fragile under quant) | Med | Mobile demo regresses | Quant ablation: keep verifier in FP16 (it's only 400M ≈ 800 MB FP16, fits on Galaxy S24+); only quant the 1.5B reasoner. If memory still tight, Q5_K_M for the verifier |
| 7 | **No physical Galaxy device** during build window | Med | Demo regression | Fallback ladder: (a) Galaxy S24 (best), (b) any Snapdragon 8 Gen 2/3 Android via ADB, (c) Termux + llama.cpp on Android, (d) `llama.cpp` on a MacBook M2/M3 with `n_gpu_layers` for an honest mobile-class CPU/Metal demo. Lead with the best available and be transparent in the report |
| 8 | **Single GPU dies** mid-run | Low | 2 day delay | Push checkpoints to HF Hub every 50 steps; have a Vast.ai / Lambda Cloud RTX 4090 snippet ready as failover; budget ~$1.50/hr × 36 h ≈ $60 contingency |
| 9 | **TTRL add-on doesn't improve numbers** | Med | Lose 1 day | Mark TTRL explicitly as "ablation add-on" not "core deliverable". If it doesn't help, report the negative result honestly — that's also a contribution |
| 10 | **Pass@k diversity collapse** under GRPO | Med | Weak ablation | Track Pass@1 *and* Pass@8 / Pass@64 throughout training; add Clip-Cov / entropy-mechanism (arxiv:2505.22617) if collapse appears; this is in our reference list anyway |
| 11 | **Reproducibility breaks** between author machine and judge machine | Low | Disqualification risk | Dockerfile alongside `requirements.txt`; CI smoke-test on a free GitHub Actions runner that at least imports everything and runs unit tests |
| 12 | **Time mismanagement** in the last 3 days (report + video + repo polish) | High | Submission incomplete | Hard freeze: code complete by end of day 11; days 12–14 are *only* report + video + polish. See `execution.md` |

---

## 6. Stage-level technical plan

### Stage 1 — Verifier (≈ 1.5 days wall, ≈ 6 h GPU)
- **Inputs:** GSM8K + MATH-500 + StrategyQA prompts (~12k); teacher = DeepSeek-V3
  (preferred) or GPT-4o; teacher prompt: "Rate difficulty 0–1 for the model
  `DeepSeek-R1-Distill-Qwen-1.5B`. 0 = trivial (single-step), 1 = needs full CoT."
  Run teacher 5 times per item, take mean → soft label.
- **Architecture:** 400M cross-encoder, init from `microsoft/MiniLM-L12-H384`
  *or* a frozen `Qwen2.5-0.5B` encoder + linear head. Match Weaver's design
  closely — they validated this exact size.
- **Loss:** MSE on soft labels + auxiliary BCE on hard threshold (`d > 0.5`).
- **Eval:** Spearman ρ vs. teacher labels on a 500-item held-out set.
  Target: ρ > 0.7. If we hit 0.6 we still proceed; below 0.5 redo with
  cleaner teacher prompt.

### Stage 2 — GRPO Router (≈ 2.5 days wall, ≈ 36 h GPU)
- **Base:** `DeepSeek-R1-Distill-Qwen-1.5B`, frozen + QLoRA-4bit adapter.
- **Routing head:** prepend a 2-token decision (`<think>` / `<no_think>`) to
  the response. Reward depends on whether the model honours its own decision
  (a constraint-respect bonus prevents Qwen3-style "no-think but still thinks").
- **Reward:**
  ```
  r = correctness                               # 0/1 from rule + verifier
      − λ_tok · n_tokens · (1 − d)              # length penalty, gated by ext. difficulty
      − λ_kl  · KL(π || π_ref)                  # standard GRPO regularisation
      + λ_obey · 1{decision honoured}           # anti-leakage bonus
  ```
- **Hyperparameters (initial):** `λ_tok = 5e-4`, `λ_kl = 5e-3`,
  `λ_obey = 0.1`, `group_size = 8`, `lr = 1e-5`, `1500 steps`, batch 4,
  context 1024+1024, temperature 0.7, vLLM colocated at
  `gpu_memory_utilization=0.45`.
- **Tracked metrics:** reward mean & components, KL, response length,
  think-rate, Pass@1, Pass@8, Pass@64 every 100 steps on a tiny eval set.
- **Sweep (if time):** `λ_tok ∈ {2e-4, 5e-4, 1e-3}`, `λ_kl ∈ {1e-3, 5e-3, 2e-2}`.

### Stage 2.5 — TTRL add-on (optional, ≈ 1 day)
Layer TTRL on top of the Stage-2 checkpoint using majority-vote pseudo-labels
on MMLU subsets. Plot a second curve on the Pareto. If it doesn't help,
include the negative result.

### Stage 3 — Quantise + on-device (≈ 1.5 days)
- Convert reasoner to GGUF Q4_K_M via `llama.cpp` `convert_hf_to_gguf.py` +
  `llama-quantize`.
- Verifier stays FP16 (or Q5_K_M if RAM-constrained) — cross-encoders are
  fragile under aggressive quant.
- `llama-cli` / `llama-server` for the reasoner, `onnxruntime-mobile` or
  `llama.cpp`'s embedding mode for the verifier.
- ADB harness: push both models, run a 200-question battery, log
  per-question latency, peak RSS, energy via `dumpsys batterystats`. Plot.

### Stage 4 — Eval, ablate, report (≈ 2.5 days)
- Full benchmark sweep: GSM8K (Pass@1, Pass@k k∈{1,8,64,256}), MMLU subset,
  StrategyQA, AIME24 (30 problems), all with 3 seeds.
- Ablation table:
  - Reward components (turn off length penalty / verifier gating / obey bonus)
  - Routing input (prompt-only / +logits / +verifier)
  - Quantisation (FP16 / Q4_K_M / Q5_K_M)
  - TTRL on/off
- Pareto chart, report writing, video shoot, repo polish.

---

## 7. What "winning" looks like — concrete success criteria

A submission earns top-tier scoring if **all** of the following hold:

- [ ] Pareto chart with our system **strictly dominating** at least one of
      {full-CoT, no-CoT} at the same accuracy *or* same token cost.
- [ ] At least **30% average token reduction** on GSM8K with **≤ 1% accuracy
      drop**, with 3-seed 95% CIs.
- [ ] Working **on-device demo on a real Galaxy** (or Snapdragon Android with
      ADB) recorded in the 3-minute video.
- [ ] **Published HF artefacts:** verifier checkpoint, router LoRA adapter,
      teacher-label dataset, all under Apache-2.0 with model/dataset cards.
- [ ] **One-command reproducibility:** `bash scripts/06_eval_all.sh` reproduces
      the headline number from a fresh checkout on a 24 GB GPU.
- [ ] **8-page report** with headline number in the title and explicit
      acknowledgement of AdaptThink / Qwen3 / Weaver / TTRL prior art.

A "stretch win" additionally has:

- [ ] TTRL add-on improves Pareto on at least one held-out benchmark.
- [ ] Battery-drain comparison (full-CoT vs. AdaptiveThink) on a real Galaxy.
- [ ] Latency under 2 s median for easy GSM8K items on Galaxy S24.

---

## 8. What we explicitly will **not** do

- ❌ Train a new RL algorithm. We use vanilla GRPO via TRL. Algorithmic novelty
  is not where we win.
- ❌ Train at long context (>2k). 1024+1024 is the sweet spot for 24 GB; chasing
  longer context loses us the run.
- ❌ Build a UI app for the phone. ADB-driven CLI demo is enough to record a
  convincing 3-minute video. A polished Android app is week 4 work.
- ❌ Compare against closed models (GPT-4o, Claude) as our baseline. Our
  baselines are the published open SLM reasoners — that's the fair comparison.
- ❌ Promise things in the report that aren't in the results CSV.

---

## 9. Roles (StateZero — 2 people)

- **Jeeth (lead):** environment + GRPO router + on-device deployment + report.
- **Ojasvi:** verifier distillation + teacher labels + evaluation harness +
  ablations + video.

Both review each other's PRs. Daily 15-minute sync at the end of each day to
update `execution.md`.

---

## 10. Single most important sentence

> **Win condition: a measured Pareto curve on a Galaxy phone, with honest
> citations and one-command reproducibility.** Every hour of work either
> moves a point on that chart, or it doesn't.
