# CLAUDE.md — Operating Instructions for AI Assistants

> Read this **first**, every session, before touching code or docs in this repo.
> This file is the contract between the human team (Jeeth, Ojasvi) and any AI
> coding assistant working on the **StateZero / AdaptiveThink** Round-2 build

---

## 1. What this repo is

**Project:** AdaptiveThink — an on-device adaptive-reasoning system for small
language models (SLMs), built around three coupled artefacts:

1. **Reasoner:** `DeepSeek-R1-Distill-Qwen-1.5B` (frozen base + QLoRA adapter).
2. **Difficulty Verifier:** a 400M Weaver-style cross-encoder, distilled from a
   teacher ensemble (DeepSeek-V3 / GPT-4o judgments).
3. **Routing Head:** a 2-token RL-trained head (`<think>` / `<no_think>`) on top
   of the reasoner, trained with GRPO and reward = `correctness − λ · tokens`.

**Final deliverable:** quantised GGUF Q4_K_M bundle running on a Samsung Galaxy
device, plus a reproducible training pipeline, technical report, and 3-min demo.

**Hackathon:** Samsung EnnovateX 2026
 — PS06 (RL for SLM Reasoning), Round 2.
We have already cleared Round 1 with the proposal in
`StateZero_06_AdaptiveThink.pdf`. We must now **build** the system.

---

## 1b. Required API keys

Before running anything, create `.env` (git-ignored) with these keys:

```
# Required for teacher label generation (pick ONE)
DEEPINFRA_API_KEY=...          # DeepSeek-V3 via DeepInfra (~$0.14/M tokens)
# OR
OPENAI_API_KEY=...             # GPT-4o fallback (~$5/M tokens, more expensive)
# OR
TOGETHER_API_KEY=              # Together AI (Qwen2.5-72B free tier fallback)

# Required for experiment tracking
WANDB_API_KEY=...              # wandb.ai → Settings → API keys

# Required for checkpoint backup
HF_TOKEN=...                   # huggingface.co → Settings → Access Tokens (write)
```

**Minimum viable set:** `DEEPINFRA_API_KEY` + `WANDB_API_KEY` + `HF_TOKEN`.
`OPENAI_API_KEY` only needed if DeepInfra is down.
Galaxy device: deferred — on-device benchmarks are Day 8+, not blocking.

---

## 2. Honest novelty position (read this before writing any docs)

The original Round-1 proposal claimed: *"First system to RL-train a separate
routing model."* **This claim is no longer true** as of May–June 2025. Concretely:

| Paper | Date | What it does | Overlap |
|---|---|---|---|
| AdaptThink (arxiv:2505.13417) | May 2025 | RL-trains think/no-think via constrained opt + importance sampling; **internal** difficulty signal only | Direct but no external verifier |
| Adaptive Thinking Mode Switching (2505.15400) | May 2025 | Accuracy-aware length reward regulation | Direct |
| Adaptive Thinking Preferences (2506.18237) | Jun 2025 | Group-relative reward using model confidence | Direct |
| Qwen3 Technical Report (2505.09388) | May 2025 | Hybrid thinking + thinking-budget mechanism | Direct |
| CODA (arxiv:2603.08659) | Mar 2026 | Difficulty-aware gates on length penalty; **policy-internal** difficulty via group rollouts | **Closest** — but internal signal, no external verifier |
| DIET (2505.19217) | May 2025 | Difficulty-conditioned token penalty | Adjacent |
| GRPO-LEAD (2504.09696) | Apr 2025 | Difficulty-aware advantage reweighting | Adjacent |
| Efficient Reasoning w/ Adaptive Length Penalties (2506.05256) | Jun 2025 | `correctness − λ·length` reward | Exact reward shape |
| R2R (2505.21600) | May 2025 | Token-level router between R1-1.5B and R1-32B | Adjacent |
| Weaver (Stanford, 2506.18203) | Jun 2025 | 400M distilled verifier, 98.7% retention | We use it |
| TTRL (NeurIPS 2025, 2504.16084) | Apr 2025 | Test-time RL with majority-vote pseudo-labels | We layer it |

**Our defensible novelty (system contribution):**

> First end-to-end, fully on-device pipeline that **composes** (a) a separately-
> distilled 400M external difficulty verifier, (b) an RL-trained 2-token
> routing head on a 1.5B reasoner, and (c) GGUF Q4_K_M deployment on a real
> Samsung Galaxy device, with a **measured** latency / accuracy Pareto curve
> on consumer mobile hardware.

Every other paper above shows one or two of these pieces in isolation. **No one
has shipped the full stack quantised onto a phone with measured numbers.** That
is what we build.

When writing the report, the demo script, or any communication: **never** say
"first to RL-train a router". Say "first end-to-end on-device adaptive-reasoning
system with external distilled verifier and measured mobile Pareto".

---

## 3. Repo layout (target — does not yet exist)

```
ENNOVATE/
├── README.md                      # public-facing, judge-readable
├── CLAUDE.md                      # this file
├── plan.md                        # strategy, novelty, risk, judging map
├── execution.md                   # day-by-day runbook
├── StateZero_06_AdaptiveThink.pdf # original Round-1 deck (do not edit)
│
├── configs/
│   ├── verifier_distill.yaml      # Stage-1 config
│   ├── grpo_router.yaml           # Stage-2 config (Unsloth + TRL + vLLM)
│   ├── ttrl_addon.yaml            # optional Stage-2.5
│   └── quantize_gguf.yaml         # Stage-3 config
│
├── src/adaptivethink/
│   ├── __init__.py
│   ├── data/                      # GSM8K / MATH-500 / StrategyQA / MMLU loaders
│   ├── verifier/                  # 400M cross-encoder + teacher-label gen
│   ├── router/                    # routing head, prompts, GRPO reward fn
│   ├── ttrl/                      # majority-vote pseudo-label TTRL
│   ├── eval/                      # Pass@1 / Pass@k, Pareto, ablations
│   └── deploy/                    # GGUF export, on-device wrapper, ADB harness
│
├── scripts/
│   ├── 01_setup.sh                # environment + pinned wheels
│   ├── 02_gen_teacher_labels.py
│   ├── 03_train_verifier.sh
│   ├── 04_train_grpo_router.sh
│   ├── 05_ttrl_addon.sh           # optional
│   ├── 06_eval_all.sh
│   ├── 07_quantize_gguf.sh
│   └── 08_galaxy_bench.sh         # ADB-driven on-device latency/RAM/battery
│
├── tests/                         # unit tests for reward fn, dataloaders, etc.
├── notebooks/                     # only for one-off plots, never logic
├── results/
│   ├── pareto/                    # the chart that wins us the comp
│   ├── ablations/                 # 3-seed runs with CIs
│   └── demo_video/
│
├── report/                        # technical report (max 8 pages)
└── requirements.txt               # pinned exact versions
```

---

## 4. Hard rules for any AI assistant

These are **non-negotiable**. Violations cost us the hackathon.

### 4.1 Code

- **Pin every dependency.** `requirements.txt` uses `==`, never `>=`. The stack
  is fragile (Unsloth × TRL × vLLM × FlashAttention all need exact compatible
  versions). Last known good as of writing:
  - `unsloth` (latest from main, pin commit hash)
  - `trl==0.12.0` or higher; verify GRPOTrainer API hasn't broken
  - `vllm` in `[0.12.0, 0.18.0]`
  - `flash-attn==2.x.x` (pre-built wheel matching CUDA)
  - `torch==2.4.x` (must match flash-attn ABI)
  - `transformers` pinned to a Unsloth-compatible release
  - `bitsandbytes` matching CUDA
  - `peft` matching trl
  - `wandb`, `datasets`, `accelerate` pinned
  - `llama-cpp-python` for GGUF inference verification on host
- **Reproducibility.** Every training script sets `torch`, `numpy`, `random`,
  `transformers` seeds. Every run logs git SHA + `pip freeze` to `wandb`.
- **Determinism caveat:** GRPO with vLLM rollouts is *not* bit-exact. Run **3
  seeds** for every reported number; report mean ± 95% CI.
- **Checkpointing:** `save_steps=50`, `save_total_limit=3`,
  `hub_strategy=every_save`, push to a private HF repo. SSH dies; tmux + auto-
  resume from `last/` directory. Never trust a single GPU box.
- **No silent fallbacks.** If `vllm` colocation fails, raise loudly. Do not
  silently fall back to HF generation — the run will explode 12 hours in.
- **No new top-level frameworks** without consultation. Stick to Unsloth + TRL.
- **Match existing style** once code exists. Until then, follow PEP 8 +
  `ruff`-clean + type hints on public functions.

### 4.2 Docs

- **Never claim novelty we cannot defend.** See §2.
- **Always cite arxiv IDs** when discussing prior work, in the form
  `arxiv:2505.13417`. The judges will check.
- **Headline number in the report title.** e.g. *"AdaptiveThink: 73% token
  reduction at −0.4% accuracy on GSM8K, on a Samsung Galaxy S24."* Pick the
  number after results are in, but the report skeleton must reserve the slot.
- **Demo video on YouTube, not Drive.** Drive links get 404'd by judges.

### 4.3 Safety / scope

- **No production data, no PII.** All datasets are public open licences.
- **No closed-weight models in the deliverable.** Teacher labels from
  DeepSeek-V3 / GPT-4o are used at distillation time only; the final shipped
  artefact is 100% open-weight.
- **Do not commit:** API keys, HF tokens, wandb keys, `.env`, `wandb/` runs,
  `outputs/` checkpoints. Add to `.gitignore` on day 1.
- **Apache-2.0** for everything we ship. Add `LICENSE`, `NOTICE`,
  `CITATION.cff`.

### 4.5 Progress documentation (mandatory)

Every **50 GRPO training steps** (or at the end of each calendar day, whichever
comes first), an AI assistant or team member **must** update the following:

1. **`execution.md` §"Lessons"** — append a timestamped bullet: what happened,
   what surprised us, what we changed.
2. **`CLAUDE.md` §5 Decision log** — append any non-trivial decision made
   (hyperparameter change, architecture swap, pivot, fallback triggered).
3. **`results/progress.md`** (create if absent) — one-line summary of current
   best numbers: `[date] seed=X step=Y GSM8K_pass1=Z think_rate=W`.

**Before starting any new session**, an AI assistant must:
1. Read `results/progress.md` to know current state.
2. Read the last 5 entries in `execution.md` §"Lessons".
3. Read the last 3 entries in `CLAUDE.md` §5.
4. Only then proceed. Never assume state from memory.

This rule exists because context windows reset. A 30-hour GRPO run that
diverged 20 hours ago is invisible to a fresh session without these docs.

- **Read `plan.md` and `execution.md` before changing anything.** If a task
  isn't in the plan, ask the human first.
- **Update `execution.md` as you go.** Tick boxes, log timestamps, log
  surprises in the "Lessons" section at the bottom.
- **One PR / commit per logical step.** Squash later if needed.
- **Tests before training.** A bug in the reward function discovered 30 hours
  into GRPO is a project-killer. Unit-test the reward, the dataloader, and the
  difficulty score scaling before launching any long run.

---

## 5. Decision log (append below)

When you (the AI) make a non-trivial decision, log it here as a short bullet
with date, decision, alternatives considered, and reason. Examples:

- *2026-05-30 — Pinned `trl==0.12.1` (not 0.13) because GRPOTrainer signature
  changed in 0.13 and Unsloth main has not caught up. Re-evaluate at week 2.*
- *2026-05-31 — Reward = correctness − 0.0005 · tokens chosen over
  correctness − 0.001 · tokens after pilot run; the larger λ collapsed all
  responses to no-think on GSM8K easy split.*

- *2026-06-03 — `lambda_tok = 3e-3` (not 5e-4 from plan.md §6 initial spec).
  Pilot observation embedded in reward.py comment: 5e-4 was too weak to
  penalise easy responses meaningfully. At 3e-3 a 300-token easy response
  (d=0.1) incurs penalty 0.81, leaving reward=0.19 for a correct answer and
  negative for >330 tokens. This is intentionally aggressive to force
  no-think routing on trivial GSM8K items. Risk: collapse on medium-difficulty
  items (d≈0.4–0.6). Mitigation: monitor think_rate in first 200 steps; if
  it drops below 20% relaunch with lambda_tok=1e-3. Sweep planned: {1e-3,
  3e-3, 5e-3} as short 600-step runs if Case B on Day 6.*

---

## 6. Quick command cheatsheet

```bash
# Environment (run once)
bash scripts/01_setup.sh

# Stage 1 — teacher labels (≈ overnight, API-bound)
python scripts/02_gen_teacher_labels.py --config configs/verifier_distill.yaml

# Stage 1 — train 400M verifier (≈ 6 h on RTX 4090)
bash scripts/03_train_verifier.sh

# Stage 2 — GRPO router (≈ 36 h)
tmux new -s grpo "bash scripts/04_train_grpo_router.sh"

# Eval everything
bash scripts/06_eval_all.sh

# Quantise to GGUF
bash scripts/07_quantize_gguf.sh

# On-device benchmark (phone connected via ADB)
bash scripts/08_galaxy_bench.sh
```

---

## 7. When in doubt

1. Re-read §2 and §4.
2. Check `plan.md` for the strategic question, `execution.md` for the
   operational one.
3. If still unclear, **stop and ask the human**. A 5-minute clarification
   beats a 30-hour wasted training run.
