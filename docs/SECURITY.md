# Security and Secrets

This project is built for distributed training with strong operational controls.
Security guidance is based on the actual runtime and test behaviors in this repo.

## Credentials

- Use environment variables for secrets, not source control.
- Supported credential names in the codebase include:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `S3_ENDPOINT_URL`
- `moe-engine/tests/test_smoke_e2e.py` validates S3 behavior using a mocked
  `boto3` environment, so real credentials are never required for unit test
  coverage.

## Checkpoint storage and integrity

- The elastic checkpointing stack writes to a local staging directory before
  mirroring to remote storage.
- `moe-engine/train.py` configures `ElasticConfig` with `local_ckpt_dir` and
  `remote_uri`.
- `moe-engine/pkg/elastic/fault_monitor.py` implements chunked streaming writes
  and attempts to use `O_DIRECT` for direct I/O when available.
- This file-level durability behavior is validated by tests such as
  `moe-engine/tests/test_elastic.py::test_local_nvme_chunked_write` and
  `moe-engine/tests/test_smoke_e2e.py`.

## Network and rendezvous security

- This repository supports both c10d and etcd rendezvous backends.
- `moe-engine/pkg/elastic/fault_monitor.py` initializes `etcd` for scale and
  falls back to `c10d` when `etcd` is unavailable.
- Production deployments should use private networking and restricted rendezvous
  endpoints to protect cluster coordination traffic.

## Runtime robustness

- The project enables safer NCCL behavior by default.
- `moe-engine/pkg/elastic/fault_monitor.py` sets:
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
  - `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30`
- These settings help detect and recover from mid-collective failures in elastic
  training.

## Observability and log hygiene

- Telemetry is structured and should not contain secrets.
- Avoid logging credential values, endpoint URLs, or access keys.
- Use logs for operational state only, not secret material.

## Vulnerability disclosure

- If you discover a security issue, open a GitHub issue and describe the
  behavior without including secrets.
- Do not post access keys, credentials, or private endpoint details publicly.

## License and compliance

- This repository is Apache 2.0 licensed. Refer to `LICENSE` for the full terms.
- The docs and code are intentionally explicit about security-sensitive
  boundaries such as local NVMe staging and remote mirror storage.

