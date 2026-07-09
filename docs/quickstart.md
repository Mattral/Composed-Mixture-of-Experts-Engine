# Quickstart

**Version:** v0.3.3  
**Last updated:** July 2026

Get from zero to a running MoE training step in under 5 minutes — no GPU required.

---

## Prerequisites

- Python ≥ 3.10
- pip ≥ 23
- 4 GB RAM (smoke config)

No CUDA, no GPU, no Docker needed for steps 1–5.

---

## Step 1 — Install

**From zip (recommended — always up to date):**

```bash
unzip moe_engine_v032_final.zip
cd moe_upgraded/moe-engine/
pip install -e ".[dev]"
```

**From git (GitHub repo may be behind the zip):**

```bash
git clone https://github.com/Mattral/Composed-Mixture-of-Experts-Engine
cd Composed-Mixture-of-Experts-Engine/moe-engine/
pip install -e ".[dev]"
```

**Verify installation:**

```bash
python scripts/cli.py info
# Expected output:
#   Python:    3.12.x
#   PyTorch:   2.x.x
#   CUDA:      not available  (or: device name if GPU present)
#   Pydantic:  2.x.x
#   moe-engine: 0.3.2
#   Configs:
#     [OK] default.yaml: H=4096 E=64
#     [OK] smoke.yaml: H=32 E=4
```

**Install pre-commit hooks (optional, recommended):**

```bash
pip install pre-commit && pre-commit install
```

---

## Step 2 — Validate configs

```bash
make validate-config
# or: python scripts/cli.py validate configs/
```

Expected output:

```
Validating 2 config file(s):

  [OK]  default.yaml    H=4096 E=64 K=2 world_size=64 dtype=bfloat16
  [OK]  smoke.yaml      H=32   E=4  K=2 world_size=1  dtype=float32

All 2 config(s) valid.
```

---

## Step 3 — Run the smoke test

```bash
make smoke
# or: python train.py --config configs/smoke.yaml --smoke
```

Expected output (CPU, fp64 reference path):

```
INFO ... step=0 loss=4.8802 mfu=0.00% | 1847 tok/s | step=8.7ms
INFO ... step=1 loss=4.8741 mfu=0.00% | 1923 tok/s | step=8.3ms
...
INFO ... Smoke complete. Checkpoint written.
```

MFU is near zero because the CPU reference path is not GPU hardware.
On T4 with Triton, expect ~0.1–0.5% MFU (T4 is an inference card).
On H100 SXM5 at production scale, expect 40–55% MFU.

Outputs in `/tmp/moe-engine/logs/step.jsonl` and `/tmp/moe-engine/ckpts/`.

---

## Step 4 — Run the test suite

```bash
make test-cpu
# or: pytest tests/ -m cpu -k "not (2rank or multiprocess or distributed_invariants)" -q
```

Expected:

```
1 xfailed, 348 passed, 1 skipped in ~20s
```

- **348 passed**
- **1 skipped**: Triton GPU path (no CUDA in this environment — expected)
- **1 xfailed**: `test_routing_quality::test_uniform_init_lower_imbalance[2]`  
  Marked `xfail` — a documented statistical edge case at seed=2, not a code
  bug. See `docs/testing.md` for the full mechanism explanation.

---

## Step 5 — Explore the CLI

```bash
# All commands
python scripts/cli.py --help

# Validate a specific config
python scripts/cli.py validate configs/smoke.yaml

# Run benchmark (CPU)
python scripts/cli.py benchmark --json /tmp/bench.json

# Environment info
python scripts/cli.py info
```

---

## Step 6 (optional) — GPU smoke on T4

If you have a T4 GPU (e.g. Google Colab free tier):

1. Open `moe-engine/notebooks/moe_engine_v032_T4_validation.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Upload the zip and run cells 0–4.

This confirms the Triton kernel compiles and runs on real GPU hardware. Expected:
- Triton kernel: ✅ compiles
- Token conservation: `violations=0/100` on CUDA
- Smoke train: MFU > 0% (Triton GPU path active)

---

## Step 7 (optional) — Multi-GPU

With 4 × GPUs available locally:

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/default.yaml --max-steps 20
```

---

## Key files for new contributors

| File | Purpose |
|------|---------|
| `configs/smoke.yaml` | Toy config (H=32, E=4) — fast development |
| `configs/default.yaml` | Production config (H=4096, E=64) — 64 GPUs |
| `pkg/distributed/` | 7 focused modules + backward-compat shim |
| `pkg/kernels/moe_router.py` | Triton fused router kernel |
| `pkg/utils/config.py` | Pydantic MoEConfig (start here for config changes) |
| `pkg/models/registry.py` | Model registry and factory |
| `tests/test_config.py` | 38 config system tests |
| `tests/test_properties.py` | Property-based invariant tests |
| `docs/ARCHITECTURE.md` | Component map with sequence diagrams |
| `docs/LIMITED_HARDWARE_GUIDE.md` | Developing without a GPU cluster |
| `docs/adr/` | Why major design decisions were made |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pkg'`**

Run from the `moe-engine/` directory, or `pip install -e ".[dev]"` first.

**`ConfigValidationError: [model] top_k (4) must be <= num_experts (4)`**

Your config has `top_k` greater than `num_experts`. Reduce `top_k` or increase
`num_experts`. The Pydantic validator catches this at load time.

**`triton.compiler.errors.CompilationError: K must be constexpr`**

This is the v0.3.1 → v0.3.2 Triton fix. Confirm you are using the v0.3.2 zip
(not the GitHub repo). Run Section 2 of the T4 notebook to verify.

**Smoke test fails with `NaN in output`**

Gate weight initialisation may be too large. Try:
```python
# In pkg/models/moe.py, reduce init scale
nn.init.normal_(self.w_gate.weight, std=0.01)
```

**Pre-commit `detect-secrets` fails on first run**

```bash
detect-secrets scan > .secrets.baseline
git add .secrets.baseline && git commit -m "chore: init secrets baseline"
```
