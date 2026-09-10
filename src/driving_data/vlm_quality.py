"""A real local VLM tagger benchmark. No network inference, no metadata in the prompt."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import digest, file_hash, image_path, read_json, write_json

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
PROMPT = "Classify the lighting in this road image. Answer with exactly one word: daytime or night."


def parse_label(text):
    value = text.strip().lower().strip(".\"' ")
    return value if value in {"daytime", "night"} else None


def summarize(rows, field):
    n = len(rows)
    if not n:
        raise ValueError("Empty audit set")
    correct = sum(r[field] == r["reference"] for r in rows)
    per_class = {}
    for label in ("daytime", "night"):
        subset = [r for r in rows if r["reference"] == label]
        per_class[label] = {
            "images": len(subset),
            "recall": sum(r[field] == label for r in subset) / len(subset) if subset else None,
        }
    return {
        "images": n,
        "correct": correct,
        "accuracy": correct / n,
        "invalid_outputs": sum(r[field] is None for r in rows),
        "classes": per_class,
        "balanced_accuracy": float(
            np.mean([v["recall"] for v in per_class.values() if v["recall"] is not None])
        ),
    }


def allowed_next(prefix, sequences, eos_token_id):
    choices = set()
    for sequence in sequences:
        if sequence[: len(prefix)] == prefix:
            choices.add(sequence[len(prefix)] if len(sequence) > len(prefix) else eos_token_id)
    if not choices:
        raise ValueError("Invalid constrained decoding prefix")
    return sorted(choices)


def vlm_quality(root: Path, constrained=False):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(4)
    manifest = read_json(root / "data/manifest.json")
    # Restrict this exploratory task to development data. Official labels are a reference, not human QA.
    rows = [r for r in manifest["rows"] if r["split"] == "dev" and r["timeofday"] in ("daytime", "night")]
    protocol = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dataset_version": manifest["version"],
        "split": "dev",
        "sample_ids": [r["sample_id"] for r in rows],
        "prompt": PROMPT,
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 8,
        "do_sample": False,
        "image_max_edge": 512,
        "rule_night_luminance_below": 70,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "scope": "Reference-label agreement for day/night tags only. "
        "No manual gold adjudication, cost savings, bbox QA or safety validation.",
    }
    if constrained:
        protocol["decoding"] = "finite label trie: daytime/night; post-hoc dev-set format intervention"
    outdir = root / ("reports/pilot/vlm_quality_constrained" if constrained else "reports/pilot/vlm_quality")
    path = outdir / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("VLM protocol changed; use a separate run")
    write_json(path, protocol)
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=root / "models/hf",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
            max_workers=2,
        )
    )
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    weight_sha = file_hash(snapshot / "model.safetensors")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        add_generation_prompt=True,
    )
    records, cached = [], 0
    started = time.perf_counter()
    for row in rows:
        if file_hash(image_path(root, row)) != row["sha256"]:
            raise ValueError("VLM input image changed")
        key = digest([protocol, row["sha256"], weight_sha])
        cache = root / "work/vlm_cache" / f"{key}.json"
        if cache.exists():
            record = read_json(cache)
            cached += 1
        else:
            with Image.open(image_path(root, row)) as original:
                im = original.convert("RGB")
                im.thumbnail((512, 512))
            tic = time.perf_counter()
            try:
                inputs = processor(text=prompt, images=[im], return_tensors="pt")
                generation_kwargs = {}
                if constrained:
                    sequences = [
                        processor.tokenizer.encode(label, add_special_tokens=False)
                        for label in ("daytime", "night")
                    ]
                    prompt_length = inputs["input_ids"].shape[1]

                    def next_tokens(_batch_id, ids):
                        return allowed_next(
                            ids[prompt_length:].tolist(), sequences, processor.tokenizer.eos_token_id
                        )

                    generation_kwargs["prefix_allowed_tokens_fn"] = next_tokens
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=8, do_sample=False, **generation_kwargs
                    )
                generated_only = generated[:, inputs["input_ids"].shape[1] :]
                raw = processor.batch_decode(generated_only, skip_special_tokens=True)[0]
                record = {
                    "sample_id": row["sample_id"],
                    "sha256": row["sha256"],
                    "raw_output": raw,
                    "vlm": parse_label(raw),
                    "error_type": None,
                    "input_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
                    "latency_seconds": time.perf_counter() - tic,
                }
            except Exception as e:
                record = {
                    "sample_id": row["sample_id"],
                    "sha256": row["sha256"],
                    "raw_output": "",
                    "vlm": None,
                    "error_type": type(e).__name__,
                    "latency_seconds": time.perf_counter() - tic,
                }
            if record["error_type"] is None:
                write_json(cache, record)
        # Join reference metadata only AFTER inference. It is never a model input.
        records.append(
            {
                **record,
                "reference": row["timeofday"],
                "rule": "night" if row["mean_luminance"] < 70 else "daytime",
                "majority": "daytime",
            }
        )
        if len(records) % 6 == 0:
            print(f"VLM {len(records)}/{len(rows)}", flush=True)
    report = {
        "protocol_sha256": digest(protocol),
        "weight_sha256": weight_sha,
        "rows": records,
        "metrics": {f: summarize(records, f) for f in ("majority", "rule", "vlm")},
        "cache_hits": cached,
        "seconds_excluding_model_load": time.perf_counter() - started,
        "median_inference_seconds": float(np.median([r["latency_seconds"] for r in records])),
        "system_errors": sum(r["error_type"] is not None for r in records),
        "review_queue": [
            {"sample_id": r["sample_id"], "reason": "invalid_or_disagreement", "status": "unreviewed"}
            for r in records
            if r["vlm"] is None or r["vlm"] != r["reference"] or r["vlm"] != r["rule"]
        ],
    }
    write_json(outdir / "results.json", report)
    return {
        "metrics": report["metrics"],
        "system_errors": report["system_errors"],
        "review_queue": len(report["review_queue"]),
    }
