# 🏭 CS336 Assignment 2 — Systems: Making Transformers Fast

Make the Assignment-1 model fast on one GPU, then spread it across many. 🔧 First profile and benchmark
where time and memory actually go, then write **FlashAttention-2 in Triton** ⚡, then climb the parallelism
ladder: DDP → optimizer-state sharding → FSDP, backing every step with hand-derived communication math.
Training on the staff A1 model in [`cs336-basics/`](./cs336-basics). Handout:
[`cs336_assignment2_systems.pdf`](./cs336_assignment2_systems.pdf).

**Progress** — ⬜⬜⬜⬜⬜⬜⬜⬜ **0 / 8 pieces** written (137 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **Profiling & benchmarking harness** · 16 pts — time fwd/bwd/optimizer step, Nsight Systems trace,
  fp16/bf16 mixed precision (+ accumulation), memory profiling 📊
- [ ] **Activation checkpointing** · 4 pts — recompute activations to trade compute for memory 🧮
- [ ] **Attention kernels** · 29 pts — PyTorch baseline, `torch.compile`, then **FlashAttention-2** in Triton
  (forward + backward), benchmarked at batch 1 ⚡
- [ ] **DDP from scratch** · 21 pts — ring all-reduce benchmarks, a naive DDP, then overlapping
  per-parameter gradient sync 🔗
- [ ] **Optimizer state sharding** · 20 pts — ZeRO-style sharding of the optimizer moments 💾
- [ ] **FSDP** · 20 pts — all-gather weights for compute, reduce-scatter gradients 🧩
- [ ] **Parallelism math** · 17 pts — alternate ring all-reduce, plus DP / FSDP / TP / 2D calculations ✍️
- [ ] **Leaderboard** · 10 pts — fastest training step on 2× B200 🏁

## 🏃 Getting it running

1. `uv sync` — deps are managed with [`uv`](https://docs.astral.sh/uv/); the local `cs336-basics` package is
   pulled in via `pyproject.toml`.
2. Sanity-check the import: `uv run python -c "import cs336_basics.model"`.
3. Wire my code into [`tests/adapters.py`](./tests/adapters.py) (glue only) and run `uv run pytest` — the
   DDP/FSDP tests need ≥2 GPUs, so debug on CPU + Gloo first 🔁

<!-- ## 📊 Completed results -->

## 🧭 Repo layout

- `cs336_systems/` — my implementation (empty right now, on purpose)
- `cs336-basics/` — staff Assignment-1 model to profile (`cs336_basics.model`, …)
- `tests/adapters.py` — glue between my code and the tests
- `tests/test_attention.py` · `test_ddp.py` · `test_fsdp.py` · `test_sharded_optimizer.py` — the suites to pass
- `pyproject.toml` · `test_and_make_submission.sh` · `cs336_assignment2_systems.pdf`

---

Fast on one GPU, then fast on all of them. ⚡

