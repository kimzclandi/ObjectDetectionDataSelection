"""Fixed exploratory decomposition of v2; never reads test labels or retrains."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from driving_data.common import digest, file_hash, read_json
from driving_data.evaluation import match_objects

OUT = Path("reports/diagnosis_v3")
INPUTS = [
    "data/manifest.json",
    "reports/pilot/baseline/predictions.json",
    "reports/failure_v2/dev_diagnosis.json",
    "reports/failure_v2/selections.json",
    "src/driving_data/evaluation.py",
    "scripts/diagnose_failure_v3.py",
]


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Evidence drift: {path}")
    else:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")


def ranks_corr(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auc(y, scores):
    pos = [s for s, v in zip(scores, y) if v]
    neg = [s for s, v in zip(scores, y) if not v]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def normalized_mean(vectors):
    if not vectors:
        raise ValueError("No other failure groups for prototype")
    p = np.asarray(vectors).mean(axis=0)
    return p / max(float(np.linalg.norm(p)), 1e-12)


def diagnosis(rows, predictions, labels):
    if not rows or any(r["split"] != "dev" for r in rows):
        raise ValueError("Only development rows allowed")
    records = []
    for r in rows:
        sid = r["sample_id"]
        p = predictions[sid]
        if p["sha256"] != r["sha256"] or digest(labels[sid]) != r["labels_sha256"]:
            raise ValueError("Image or reference identity drift")
        hi = match_objects(labels[sid], p["predictions"], 0.5)
        lo = match_objects(labels[sid], p["predictions"], 0.05)
        recover = lo["tp"] - hi["tp"]
        low = [x for x in p["predictions"] if 0.05 <= x["score"] < 0.5]
        records.append(
            dict(
                sample_id=sid,
                group_id=r["group_id"],
                objects=len(labels[sid]),
                recoverable=recover,
                low_count=len(low),
                low_fp=lo["fp"] - hi["fp"],
                unrecovered=lo["fn"],
                mass=sum(x["score"] * (1 - x["score"]) for x in low),
            )
        )
    for r in records:
        # Excludes the full group, including non-failure group members.
        others = [x for x in records if x["recoverable"] > 0 and x["group_id"] != r["group_id"]]
        proto = normalized_mean([predictions[x["sample_id"]]["embedding"] for x in others])
        r["prototype_groups"] = sorted({x["group_id"] for x in others})
        r["similarity"] = max(0.0, float(np.dot(predictions[r["sample_id"]]["embedding"], proto)))
        r["score"] = r["similarity"] * math.log1p(r["mass"])
    return records


def identity():
    manifest = read_json(Path("data/manifest.json"))
    return dict(
        files={p: file_hash(Path(p)) for p in INPUTS},
        dev_labels={
            r["sample_id"]: file_hash(Path(f"data/labels/{r['sample_id']}.json"))
            for r in manifest["rows"]
            if r["split"] == "dev"
        },
    )


def compute():
    manifest = read_json(Path("data/manifest.json"))
    preds = {r["sample_id"]: r for r in read_json(Path(INPUTS[1]))["rows"]}
    dev = [r for r in manifest["rows"] if r["split"] == "dev"]
    labels = {r["sample_id"]: read_json(Path(f"data/labels/{r['sample_id']}.json")) for r in dev}
    records = diagnosis(dev, preds, labels)
    ranking = read_json(Path(INPUTS[3]))["ranking"]
    components = {
        "full": {r["sample_id"]: r["score"] for r in ranking},
        "mass": {r["sample_id"]: r["low_confidence_mass"] for r in ranking},
        "similarity": {r["sample_id"]: r["dev_failure_similarity"] for r in ranking},
        "low_count": {
            r["sample_id"]: sum(0.05 <= x["score"] < 0.5 for x in preds[r["sample_id"]]["predictions"])
            for r in ranking
        },
    }
    order = {k: sorted(v, key=lambda sid: (-v[sid], sid)) for k, v in components.items()}
    ids = sorted(components["full"])
    stats = {
        k: dict(
            spearman_full=ranks_corr([components["full"][i] for i in ids], [v[i] for i in ids]),
            top36_overlap=len(set(order["full"][:36]) & set(order[k][:36])),
            min=min(v.values()),
            max=max(v.values()),
            std=float(np.std(list(v.values()))),
        )
        for k, v in components.items()
    }
    y = [r["recoverable"] > 0 for r in records]
    return dict(
        scope="Exploratory fixed dev diagnosis; pool ranking inputs only; no test access, no training or strategy selection",
        dev_records=records,
        pool_components=components,
        pool_stats=stats,
        dev_auc={k: auc(y, [r[k] for r in records]) for k in ("similarity", "mass", "low_count", "score")},
        totals={
            k: sum(r[k] for r in records)
            for k in ("objects", "recoverable", "low_count", "low_fp", "unrecovered")
        },
        limitations=[
            "LOO prototypes share training members; AUC descriptive, not independent validation.",
            "Unmatched boxes are reference-relative FP, not confirmed background or label errors.",
            "Low-threshold misses do not prove absence of RPN proposals or irreparability by ROI head.",
            "No causal attribution of v2 AP difference from these associations.",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    if args.action == "freeze":
        immutable(
            OUT / "protocol.json",
            dict(
                identity=identity(),
                design="One exploratory decomposition. Pool: full/mass/similarity/count Spearman and top36 overlap. Dev: group-excluded failure centroid AUC, low-score matching FP fraction and residual misses. No threshold search, test labels or retraining.",
                statistical_unit="development image/group; descriptive finite sample; no significance claim",
                decision="Use mechanism diagnostics to decide whether a future separately frozen confirmation is warranted; do not retrofit v2.",
            ),
        )
        return
    protocol = read_json(OUT / "protocol.json")
    if protocol["identity"] != identity():
        raise ValueError("Frozen diagnosis identity drift")
    result = compute()
    if args.action == "verify":
        if result != read_json(OUT / "result.json"):
            raise ValueError("Metric replay mismatch")
        print("Verified diagnosis from fixed raw development predictions and reference labels")
    else:
        immutable(OUT / "result.json", result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("dev_records", "pool_components")}, indent=2
            )
        )


if __name__ == "__main__":
    main()
