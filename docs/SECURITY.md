# Security and Secrets

**Version:** v0.2  
**Last updated:** June 2026

This document describes the security boundaries of moe-engine and the
controls implemented in the codebase. All guidance is grounded in the
actual runtime and test behaviour.

---

## Credentials and Secrets

### Rule: environment variables only, never source

The codebase reads credentials from environment variables at runtime.
No credentials are ever written to config files, telemetry output, or logs.

Supported credential environment variables:

| Variable | Purpose | Used in |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | S3/MinIO access key | `pkg/elastic/fault_monitor.py` (`S3Adapter`) |
| `AWS_SECRET_ACCESS_KEY` | S3/MinIO secret | `pkg/elastic/fault_monitor.py` (`S3Adapter`) |
| `S3_ENDPOINT_URL` | Custom S3/MinIO endpoint | `pkg/elastic/fault_monitor.py`, `train.py` |

**Test coverage:** `tests/test_smoke_e2e.py` validates S3 behaviour using
`moto` (boto3 mock). Real credentials are never required to run the test suite.

### Kubernetes secrets

In Kubernetes deployments, credentials are injected via a `Secret` resource:

```bash
kubectl create secret generic moe-engine-s3 -n moe-engine \
  --from-literal=endpoint=http://minio.internal:9000 \
  --from-literal=access_key_id=<key> \
  --from-literal=secret_access_key=<secret>
```

The Job specs in `deploy/k8s/` read these via `secretKeyRef`. If the secret
does not exist, the S3 keys are `optional: true` and local-only checkpointing
is used. This means the job starts successfully even without S3 credentials —
but checkpoints will not survive node replacement.

---

## Checkpoint Integrity

### Atomic writes prevent partial reads

Every checkpoint shard is written as a temporary file
(`tmp_step=<N>_rank=<R>.pt`) and renamed to its final path
(`step=<N>/rank=<R>.pt`) atomically. Rename is atomic on POSIX filesystems.
A power failure mid-write leaves only the `.tmp` file, which is ignored on
resume. This guarantees: every checkpoint shard is either fully present or
absent — no partial reads.

**Test coverage:** `tests/test_elastic.py::test_local_nvme_chunked_write`
and `tests/test_elastic_v02.py::test_file_uri_remote_tier` validate this.

### O_DIRECT for NVMe durability

`LocalNVMeAdapter.put()` attempts to open files with `O_DIRECT` (Linux only).
`O_DIRECT` bypasses the page cache: writes are sent directly to the storage
controller in the order issued, without OS buffering. This eliminates the risk
of writes being silently reordered or deferred by the kernel.

If `O_DIRECT` is unavailable (non-Linux, non-root, or unsupported filesystem),
the adapter falls back to standard buffered I/O with a warning log.

### Checkpoint metadata integrity

Every shard write produces a `.meta.json` file with:

```json
{
  "step": 500,
  "rank": 0,
  "ts": 1748901234.56,
  "hostname": "gpu-node-07.internal"
}
```

This allows detection of mismatched step numbers across ranks (which would
indicate a partial checkpoint set) and verification that a checkpoint was
produced by the expected host.

---

## Network and Rendezvous Security

### Private networking required for production

All cluster communication (NCCL collectives, Gloo rendezvous, etcd) should
occur over a private network. moe-engine does not encrypt inter-process
communication — NCCL and Gloo transmit data in plaintext over the configured
network interface.

In Kubernetes, use NetworkPolicy to restrict traffic to the training namespace:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: moe-engine-isolation
  namespace: moe-engine
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: moe-engine
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: moe-engine
```

### Rendezvous endpoint access

The rendezvous endpoint (`RDZV_ENDPOINT`) must be accessible to all nodes but
should not be exposed to the public internet. An attacker who can connect to
the rendezvous endpoint can inject fake rank registrations, disrupting training.

For c10d (`RDZV_ENDPOINT=head-node:29500`): restrict port 29500 to the training
subnet at the firewall level.

For etcd (`RDZV_ENDPOINT=etcd.internal:2379`): restrict etcd client port 2379
to the training namespace. Use etcd's built-in authentication if your etcd
cluster is shared.

### NCCL async error handling

The runtime sets two NCCL safety variables unconditionally:

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
```

`ASYNC_ERROR_HANDLING=1` aborts the process on any NCCL collective error rather
than hanging. `HEARTBEAT_TIMEOUT_SEC=30` detects stuck collectives (e.g., from
a dead peer) within 30 seconds. Both are set in `pkg/elastic/fault_monitor.py`
and in the Docker image via the `ENV` directive.

---

## Telemetry and Log Hygiene

### Telemetry must not contain secrets

The JSONL telemetry output includes only operational metrics:
step, loss, MFU, timing, memory, routing quality. It must never contain:
- Credentials or access keys
- Endpoint URLs that reveal network topology
- Model weights or activation values

This is not enforced automatically — it requires discipline in any custom
telemetry additions. The `StepRecord` dataclass defines the canonical fields;
do not add free-form string fields that could inadvertently capture config values.

### Log verbosity

`logging.INFO` is the default level in `train.py`. The only secrets that
could plausibly appear in logs are S3 endpoint URLs if they are logged at
`DEBUG` level during boto3 request setup. Set `BOTO_LOG_LEVEL=WARNING` in
production to suppress boto3 debug output.

---

## Docker Image Security

The multi-stage `Dockerfile` (`deploy/docker/Dockerfile`):

- Uses PyTorch official base images (from `pytorch/pytorch`).
- Copies source as read-only in the runtime stage.
- Does not embed any credentials or endpoints.
- Does not run as root (inherits the PyTorch base image user, typically `root`
  for GPU images — override with `USER` directive for hardened deployments).

For hardened deployments, extend the Dockerfile:

```dockerfile
FROM moe-engine:v0.2
RUN useradd -m -u 1000 trainer
USER trainer
```

---

## Vulnerability Disclosure

If you discover a security issue:

1. Open a GitHub issue with a clear description of the behaviour.
2. Include reproduction steps and the affected version.
3. Do **not** include credentials, private endpoint URLs, or secret material.
4. Do **not** post exploit code in the public issue.

We will acknowledge within 72 hours and coordinate a fix.

---

## License and Compliance

Apache 2.0. See `LICENSE` for full terms.

The Apache 2.0 license permits use in proprietary systems. It requires:
- Preservation of copyright and license notices.
- Documentation of any modifications to the source.

No export-controlled cryptography is used in this codebase. All encryption is
handled by the underlying infrastructure (HTTPS for S3, TLS for etcd) via
standard OS libraries.
