# Inference cache maintenance — 2026-09-21

Cached and newly computed predictions now require matching sample/image identities, valid detection boxes and scores, finite nonempty embeddings, and finite nonnegative latency. Corrupt cached results fail visibly instead of entering downstream selection. Existing prediction files remain unchanged on failure; no cache is silently deleted.

A run name containing predictions cannot be reused with different dataset, split or model/config identity. Choose a new `--name` for a new experiment. Valid same-identity cache hits and interrupted-task recovery remain supported. This does not claim exactly-once task execution.

The original model source is retained byte-for-byte in `baseline/` and verified against both frozen manifests. Current source is covered by regression tests. Frozen predictions, protocols and metrics are unchanged; this maintenance adds no model-quality claim.

Validation: `tests/test_inference_cache.py` exercises SQLite cache corruption, renamed experiments, valid cache reuse, interruption after task claim, and invalid newly computed output. See `validation.json` for completed checks.
