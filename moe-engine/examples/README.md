# Examples

**Version:** v0.3.2  

Self-contained, runnable examples for moe-engine. Each can be run from
`moe-engine/` with `pip install -e ".[dev]"`.

| File | Description | Time | GPU? |
|------|-------------|------|------|
| `01_router_kernel.py` | Triton router kernel correctness + throughput | ~3s | No |
| `02_moe_layer.py` | Full `DistributedMoELayer` forward + backward | ~5s | No |
| `03_config_system.py` | Pydantic `MoEConfig` usage patterns | ~1s | No |
| `04_model_registry.py` | Register and build custom models | ~2s | No |
| `05_telemetry.py` | Structured logging and step records | ~2s | No |
| `06_checkpoint.py` | Async two-tier checkpointing | ~5s | No |
| `07_custom_expert.py` | Implement and register a custom expert FFN | ~5s | No |

Run all:

```bash
cd moe-engine/
for f in examples/0*.py; do echo "=== $f ==="; python $f; done
```
