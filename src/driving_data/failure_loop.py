"""Failure-prototype selection, matched-step training, fresh test, immutable evidence."""

from __future__ import annotations

import argparse
import collections
import fcntl
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import digest, file_hash, read_json, write_json
from .data import BASE, INDEX_HASH, REVISION, convert_labels, download, group_id, validate_partition
from .evaluation import coco_metrics, match_objects
from .model import load_model, predict_one, tensor_image

OUT = Path("reports/failure_v2")
SEEDS = [101, 211, 307]
FIELDS = {"sample_id", "group_id", "split", "image_path", "sha256"}


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable evidence changed: {path}")
    else:
        write_json(path, value)


def code_identity():
    return {str(p): file_hash(p) for p in sorted(Path("src/driving_data").glob("*.py"))}


def thumbnail(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((17, 16)), dtype=np.int16)
        return a[:, :-1], a[:, 1:] > a[:, :-1]


def near_duplicate(a, b):
    return int(np.count_nonzero(a[1] != b[1])) <= 8 and float(np.abs(a[0] - b[0]).mean()) < 5


def prepare():
    old = read_json(Path("data/manifest.json"))
    index = Path("data/raw/samples.json")
    if file_hash(index) != INDEX_HASH:
        raise ValueError("Source index drift")
    groups = {r["group_id"] for r in old["rows"]}
    oldhashes = {r["sha256"] for r in old["rows"]}
    images = [(r["sample_id"], thumbnail(Path(r["image_path"]))) for r in old["rows"]]
    candidates = sorted(read_json(index)["samples"], key=lambda s: digest(["fresh-test-v2", s["filepath"]]))
    rows, excluded = [], []
    for s in candidates:
        sid = Path(s["filepath"]).stem
        gid = group_id(sid)
        if gid in groups:
            continue
        path = Path(f"data/images/{sid}.jpg")
        download(f"{BASE}/data/{sid}.jpg", path)
        thumb = thumbnail(path)
        neighbors = [name for name, t in images if near_duplicate(thumb, t)]
        sha = file_hash(path)
        if neighbors or sha in oldhashes:
            excluded.append(dict(sample_id=sid, reason="pixel_or_near_duplicate", neighbors=neighbors))
            continue
        groups.add(gid)
        images.append((sid, thumb))
        labels = convert_labels(s)
        immutable(Path(f"data/labels/{sid}.json"), labels)
        rows.append(
            dict(
                sample_id=sid,
                group_id=gid,
                split="test",
                image_path=str(path),
                sha256=sha,
                labels_sha256=digest(labels),
                width=s["metadata"]["width"],
                height=s["metadata"]["height"],
                source_revision=REVISION,
                timeofday=s.get("timeofday", {}).get("label", "unknown"),
            )
        )
        if len(rows) % 20 == 0:
            print(f"fresh-test download {len(rows)}/120", flush=True)
        if len(rows) == 120:
            break
    if len(rows) != 120:
        raise ValueError("Insufficient fresh images")
    combined = [r for r in old["rows"] if r["split"] != "test"] + rows
    validate_partition(combined)
    immutable(OUT / "manifest.json", dict(rows=combined, version=digest(combined)))
    immutable(
        OUT / "data_audit.json",
        dict(
            old_manifest=old["version"],
            new_test=120,
            excluded=excluded,
            rule="ID-hash ordered unused prefix groups; reject exact bytes or 256-bit dHash distance <=8 AND gray thumbnail MAE<5; no label-driven inclusion",
            limitation="Prefix is not verified route/session. Thumbnail near-duplicate screening is heuristic. Same upstream validation distribution, not external benchmark.",
            source_revision=REVISION,
            index_sha256=INDEX_HASH,
        ),
    )


def diagnose():
    old = read_json(Path("data/manifest.json"))["rows"]
    preds = {r["sample_id"]: r for r in read_json(Path("reports/pilot/baseline/predictions.json"))["rows"]}
    evidence = []
    vectors = []
    for r in old:
        if r["split"] != "dev":
            continue
        gold = read_json(Path(f"data/labels/{r['sample_id']}.json"))
        p = preds[r["sample_id"]]
        high = match_objects(gold, p["predictions"], 0.5)
        low = match_objects(gold, p["predictions"], 0.05)
        recover = sorted(set(low["matched_indices"]) - set(high["matched_indices"]))
        evidence.append(
            dict(
                sample_id=r["sample_id"],
                group_id=r["group_id"],
                objects=len(gold),
                tp_high=high["tp"],
                tp_low=low["tp"],
                fp_high=high["fp"],
                fp_low=low["fp"],
                recoverable_indices=recover,
                symptom="misses at score .5 with class-correct IoU>=.5 matches at score .05",
                hypotheses=[
                    "head confidence/domain mismatch",
                    "ambiguous or incomplete annotation",
                    "small-scale frozen feature limitation",
                ],
                intervention="threshold .5 -> .05, same frozen boxes; diagnostic only, does not improve AP",
                validation="Observed match recovery; causality and label correctness unresolved; no human review completed",
                decision="Use dev failure embeddings plus pool low-confidence mass for one fixed data-selection experiment; do not assume absent proposals are fixable by head training",
            )
        )
        if recover:
            vectors.append(p["embedding"])
    prototype = np.asarray(vectors).mean(axis=0)
    prototype /= max(np.linalg.norm(prototype), 1e-12)
    immutable(
        OUT / "dev_diagnosis.json",
        dict(
            records=evidence,
            prototype=prototype.tolist(),
            prototype_n=len(vectors),
            source_prediction_sha256=file_hash(Path("reports/pilot/baseline/predictions.json")),
            source_dev_labels={r["sample_id"]: r["labels_sha256"] for r in old if r["split"] == "dev"},
        ),
    )


def rank_pool(pool, predictions, prototype):
    if any(set(r) != FIELDS or r["split"] != "pool" for r in pool):
        raise ValueError("Unlabeled pool contract violated")
    proto = np.asarray(prototype)
    ranked = []
    for r in pool:
        p = predictions[r["sample_id"]]
        if p["sha256"] != r["sha256"]:
            raise ValueError("Pool prediction identity")
        mass = sum(x["score"] * (1 - x["score"]) for x in p["predictions"] if 0.05 <= x["score"] < 0.5)
        similarity = max(0.0, float(np.dot(p["embedding"], proto)))
        ranked.append(
            dict(
                sample_id=r["sample_id"],
                group_id=r["group_id"],
                low_confidence_mass=mass,
                dev_failure_similarity=similarity,
                score=similarity * math.log1p(mass),
                image_sha256=r["sha256"],
                selection_inputs="pixels, frozen predictions and dev-only failure prototype",
            )
        )
    return sorted(ranked, key=lambda r: (-r["score"], r["sample_id"]))


def select(ranked, strategy, seed, budget=12):
    candidates = ranked[:36] if strategy == "targeted" else sorted(ranked, key=lambda r: r["sample_id"])
    rng = random.Random(seed)
    order = rng.sample(candidates, len(candidates))
    selected = []
    groups = set()
    for r in order:
        if r["group_id"] not in groups:
            selected.append(r)
            groups.add(r["group_id"])
        if len(selected) == budget:
            return selected
    raise ValueError("Insufficient unique pool groups")


def freeze():
    if (OUT / "protocol.json").exists():
        raise ValueError("Protocol already frozen")
    manifest = read_json(OUT / "manifest.json")
    diagnosis = read_json(OUT / "dev_diagnosis.json")
    pool = [{k: r[k] for k in FIELDS} for r in manifest["rows"] if r["split"] == "pool"]
    preds = {r["sample_id"]: r for r in read_json(Path("reports/pilot/baseline/predictions.json"))["rows"]}
    ranked = rank_pool(pool, preds, diagnosis["prototype"])
    plans = [
        dict(strategy=s, seed=seed, selected=select(ranked, s, seed))
        for seed in SEEDS
        for s in ("targeted", "random")
    ]
    immutable(
        OUT / "selections.json",
        dict(
            ranking=ranked,
            plans=plans,
            oracle=False,
            label_access="Only after these IDs frozen; public annotations simulate acquisition",
        ),
    )
    immutable(
        OUT / "protocol.json",
        dict(
            frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            hypothesis="Dev images with recoverable low-confidence misses define a feature prototype. Labeling pool images similar to these failures with high low-confidence mass improves fresh-test AP vs random at fixed steps.",
            manifest=digest(manifest),
            diagnosis=digest(diagnosis),
            selections=file_hash(OUT / "selections.json"),
            initial_weights=file_hash(Path("models/warmup.pth")),
            code=code_identity(),
            config=read_json(Path("configs/pilot.json")),
            seeds=SEEDS,
            budget_images=12,
            steps=112,
            arms=["targeted", "random", "seed_only"],
            primary="mean paired fresh-test COCO AP(targeted-random) across all three seeds",
            secondary="AP(random-seed_only); fixed-baseline low-score-recoverable target recall, complement recall, night/day recall, precision, compute",
            selection="rank cosine(dev failure centroid, pool embedding)*log1p(sum p*(1-p) for .05<=p<.5); seeded sample 12 of top36, one per group; random uniform pool",
            training="44 seed +12 selected images, two shuffled epochs; control 44 seed +12 repeated seed slots; fresh optimizer same LR/momentum, all 112 steps; no early stop/checkpoint selection",
            evaluation="120 fresh unused prefix groups, excluded all 240 old groups, before inference frozen; no evaluations until ALL nine checkpoints complete",
            statistics="Paired prefix-group bootstrap 200 replicates for mean AP difference across seeds; same sampled groups across models/seeds; fixed seed 911204; descriptive, not population safety inference",
            decision="No stable advantage unless primary CI excludes zero positively; report all seeds/slices, no further tuning",
            environment={
                name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                for name in ("torch", "torchvision", "numpy", "Pillow", "pycocotools")
            },
            stopping="One complete comparison; retain failures and stop on corruption; never select best checkpoint/seed",
        ),
    )


def validate():
    protocol = read_json(OUT / "protocol.json")
    manifest = read_json(OUT / "manifest.json")
    if (
        protocol["code"] != code_identity()
        or protocol["manifest"] != digest(manifest)
        or protocol["initial_weights"] != file_hash(Path("models/warmup.pth"))
        or protocol["selections"] != file_hash(OUT / "selections.json")
        or protocol["diagnosis"] != digest(read_json(OUT / "dev_diagnosis.json"))
    ):
        raise ValueError("Frozen protocol drift")
    import importlib.metadata

    if protocol["environment"] != {n: importlib.metadata.version(n) for n in protocol["environment"]}:
        raise ValueError("Dependency drift")
    for r in manifest["rows"]:
        if (
            file_hash(Path(r["image_path"])) != r["sha256"]
            or digest(read_json(Path(f"data/labels/{r['sample_id']}.json"))) != r["labels_sha256"]
        ):
            raise ValueError("Image/label drift")
    return protocol, manifest


def schedule(seed_rows, selected, seed):
    if any(r["split"] != "seed" for r in seed_rows) or any(r["split"] != "pool" for r in selected):
        raise ValueError("Evaluation leaked into training")
    rng = random.Random(seed)
    slots = seed_rows + (selected if selected else rng.sample(seed_rows, 12))
    if len(slots) != 56:
        raise ValueError("Fixed 56 slots required")
    order = []
    for _ in range(2):
        epoch = list(slots)
        rng.shuffle(epoch)
        order.extend(epoch)
    return order


def train_run(name, order, seed, protocol, stop_after=None):
    out = OUT / "runs" / name
    checkpoint = Path("models/failure_v2") / f"{name}.pth"
    receipt = out / "training.json"
    recipe = dict(protocol=digest(protocol), ids=[r["sample_id"] for r in order], seed=seed)
    if receipt.exists():
        saved = read_json(receipt)
        if saved["recipe"] != recipe or file_hash(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError("Changed completed training")
        return checkpoint
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model, _ = load_model(Path("."), Path("models/warmup.pth"), "cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.roi_heads.box_predictor.parameters():
        p.requires_grad_(True)
    model.train()
    model.backbone.eval()
    cfg = protocol["config"]
    opt = torch.optim.SGD(
        model.roi_heads.box_predictor.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    statepath = Path("work/failure_v2") / f"{name}-resume.pth"
    statepath.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    elapsed = 0.0
    resumes = 0
    if statepath.exists():
        saved = torch.load(statepath, weights_only=True)
        if saved["recipe"] != recipe:
            raise ValueError("Resume recipe drift")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        losses = saved["losses"]
        elapsed = saved["seconds"]
        resumes = saved["resumes"] + 1
    started = time.perf_counter()
    for step in range(len(losses), len(order)):
        r = order[step]
        gold = read_json(Path(f"data/labels/{r['sample_id']}.json"))
        if digest(gold) != r["labels_sha256"] or file_hash(Path(r["image_path"])) != r["sha256"]:
            raise ValueError("Training input drift")
        target = dict(
            boxes=torch.tensor([g["box"] for g in gold], dtype=torch.float32).reshape(-1, 4),
            labels=torch.tensor([g["label"] for g in gold], dtype=torch.int64),
        )
        terms = model([tensor_image(Path("."), r, "cpu")], [target])
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 14 == 0 or step + 1 == len(order) or (stop_after and step + 1 == stop_after):
            tmp = statepath.with_suffix(".partial")
            torch.save(
                dict(
                    recipe=recipe,
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    rng=torch.get_rng_state(),
                    losses=losses,
                    seconds=elapsed + time.perf_counter() - started,
                    resumes=resumes,
                ),
                tmp,
            )
            os.replace(tmp, statepath)
        if stop_after and step + 1 == stop_after:
            print("Injected clean interruption after durable optimizer/RNG checkpoint", flush=True)
            return None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".partial")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, checkpoint)
    immutable(
        receipt,
        dict(
            recipe=recipe,
            checkpoint_sha256=file_hash(checkpoint),
            steps=len(losses),
            losses=losses,
            seconds=elapsed + time.perf_counter() - started,
            resumes=resumes,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            initial_weights=protocol["initial_weights"],
            label_hashes={r["sample_id"]: r["labels_sha256"] for r in order},
        ),
    )
    print(f"completed training {name}", flush=True)
    return checkpoint


def predictions(name, checkpoint, rows, protocol):
    directory = OUT / "runs" / name / "predictions"
    model, config = load_model(Path("."), checkpoint, "cpu")
    result = {}
    for r in rows:
        path = directory / f"{r['sample_id']}.json"
        key = digest([digest(protocol), config, r["sample_id"], r["sha256"]])
        if path.exists():
            record = read_json(path)
            if (
                record["key"] != key
                or record["prediction"]["sha256"] != r["sha256"]
                or record["prediction"]["sample_id"] != r["sample_id"]
            ):
                raise ValueError("Corrupted inference cache")
        else:
            try:
                pred = predict_one(model, Path("."), r, "cpu")
            except Exception as e:
                immutable(
                    OUT / "errors" / f"{name}-{r['sample_id']}-{time.time_ns()}.json",
                    dict(sample_id=r["sample_id"], error=type(e).__name__, key=key),
                )
                raise
            record = dict(key=key, prediction=pred, model=config)
            immutable(path, record)
        result[r["sample_id"]] = record["prediction"]
    if len(result) != len(rows):
        raise ValueError("Missing inference rows")
    return result


def metrics(rows, labels, preds, baseline):
    if set(preds) != {r["sample_id"] for r in rows}:
        raise ValueError("Evaluation denominator mismatch")
    counts = collections.defaultdict(collections.Counter)
    per = []
    for r in rows:
        sid = r["sample_id"]
        gold = labels[sid]
        base = baseline[sid]["predictions"]
        target = set(match_objects(gold, base, 0.05)["matched_indices"]) - set(
            match_objects(gold, base, 0.5)["matched_indices"]
        )
        m = match_objects(gold, preds[sid]["predictions"])
        matched = set(m["matched_indices"])
        parts = {
            "target": target,
            "complement": set(range(len(gold))) - target,
            "all": set(range(len(gold))),
            f"time:{r['timeofday']}": set(range(len(gold))),
        }
        detail = {}
        for tag, indices in parts.items():
            detail[tag] = dict(objects=len(indices), tp=len(indices & matched))
            counts[tag].update(detail[tag])
        per.append(
            dict(sample_id=sid, group_id=r["group_id"], slices=detail, tp=m["tp"], fp=m["fp"], fn=m["fn"])
        )
    for c in counts.values():
        c["recall"] = c["tp"] / c["objects"] if c["objects"] else None
    return dict(
        coco=coco_metrics(rows, labels, preds),
        slices=dict(counts),
        samples=per,
        precision=sum(x["tp"] for x in per) / max(1, sum(x["tp"] + x["fp"] for x in per)),
        images=len(rows),
    )


def run(interrupt=False):
    protocol, manifest = validate()
    rows = manifest["rows"]
    byid = {r["sample_id"]: r for r in rows}
    seedrows = [r for r in rows if r["split"] == "seed"]
    plans = read_json(OUT / "selections.json")["plans"]
    with (OUT / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            for arm in ("targeted", "random", "seed_only"):
                selected = (
                    []
                    if arm == "seed_only"
                    else [
                        byid[r["sample_id"]]
                        for r in next(p for p in plans if p["seed"] == seed and p["strategy"] == arm)[
                            "selected"
                        ]
                    ]
                )
                path = train_run(
                    f"{arm}-s{seed}",
                    schedule(seedrows, selected, seed),
                    seed,
                    protocol,
                    14 if interrupt else None,
                )
                if path is None:
                    return
        # Independent test labels are used for metrics only after all models are trained.
        test = [r for r in rows if r["split"] == "test"]
        baseline = predictions("baseline", Path("models/warmup.pth"), test, protocol)
        labels = {r["sample_id"]: read_json(Path(f"data/labels/{r['sample_id']}.json")) for r in test}
        immutable(OUT / "runs/baseline/evaluation.json", metrics(test, labels, baseline, baseline))
        for seed in SEEDS:
            for arm in ("targeted", "random", "seed_only"):
                name = f"{arm}-s{seed}"
                pred = predictions(name, Path(f"models/failure_v2/{name}.pth"), test, protocol)
                immutable(OUT / "runs" / name / "evaluation.json", metrics(test, labels, pred, baseline))
                print(f"completed evaluation {name}", flush=True)


def analyze(bootstrap=200):
    protocol, manifest = validate()
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    labels = {r["sample_id"]: read_json(Path(f"data/labels/{r['sample_id']}.json")) for r in rows}
    allpreds = {}
    records = []
    for arm in ("targeted", "random", "seed_only"):
        for seed in SEEDS:
            name = f"{arm}-s{seed}"
            preds = {
                p["prediction"]["sample_id"]: p["prediction"]
                for p in (read_json(f) for f in (OUT / "runs" / name / "predictions").glob("*.json"))
            }
            if len(preds) != len(rows):
                raise ValueError("Incomplete predictions")
            allpreds[name] = preds
            value = read_json(OUT / "runs" / name / "evaluation.json")
            if coco_metrics(rows, labels, preds) != value["coco"]:
                raise ValueError("Metric recomputation mismatch")
            receipt = read_json(OUT / "runs" / name / "training.json")
            records.append(
                dict(
                    name=name,
                    arm=arm,
                    seed=seed,
                    AP=value["coco"]["AP"],
                    slices=value["slices"],
                    precision=value["precision"],
                    seconds=receipt["seconds"],
                    steps=receipt["steps"],
                )
            )
    rng = random.Random(911204)
    groups = sorted({r["group_id"] for r in rows})
    boot = []
    for b in range(bootstrap):
        selected = rng.choices(groups, k=len(groups))
        sampled = [r for g in selected for r in rows if r["group_id"] == g]
        values = []
        for seed in SEEDS:
            values.append(
                coco_metrics(sampled, labels, allpreds[f"targeted-s{seed}"])["AP"]
                - coco_metrics(sampled, labels, allpreds[f"random-s{seed}"])["AP"]
            )
        boot.append(float(np.mean(values)))
        if b % 20 == 0:
            print(f"cluster bootstrap {b}/{bootstrap}", flush=True)
    deltas = [
        next(r["AP"] for r in records if r["arm"] == "targeted" and r["seed"] == s)
        - next(r["AP"] for r in records if r["arm"] == "random" and r["seed"] == s)
        for s in SEEDS
    ]
    immutable(
        OUT / "summary.json",
        dict(
            records=records,
            paired_AP_deltas=deltas,
            mean_AP_delta=float(np.mean(deltas)),
            group_bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            bootstrap_replicates=bootstrap,
            bootstrap_seed=911204,
            bootstrap_values=boot,
            protocol=digest(protocol),
            test_images=len(rows),
            test_groups=len(groups),
            conclusion="Descriptive small same-source study; no annotation cost, full detector, safety or production claim",
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "diagnose", "freeze", "run", "analyze"])
    p.add_argument("--interrupt", action="store_true")
    a = p.parse_args()
    if a.action == "run":
        run(a.interrupt)
    else:
        globals()[a.action]()


if __name__ == "__main__":
    main()
