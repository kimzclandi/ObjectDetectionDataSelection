"""Run a real subprocess interruption and recovery against the inference journal.

Only image bytes, model loading and prediction are synthetic fixtures. The
actual model.infer path, OS runner lock, SQLite journal and artifact writes run.
No neural inference, benchmark result or frozen report is produced.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from driving_data.common import file_hash, write_json  # noqa: E402


def setup(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sid in ("a", "b", "c"):
        path = output / "data/images" / f"{sid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic bytes for {sid}; no image decoder used".encode())
        rows.append({"sample_id": sid, "sha256": file_hash(path),
                     "image_path": f"data/images/{sid}.jpg", "split": "dev"})
    write_json(output / "data/manifest.json", {"version": "synthetic-recovery-fixture-v1", "rows": rows})


def child(output, phase):
    from driving_data import model

    model.load_model = lambda *_args, **_kwargs: (None, {"weight_sha256": "synthetic-model-v1"})

    def synthetic_predict(_model, _root, row, _device):
        return {"sample_id": row["sample_id"], "sha256": row["sha256"],
                "predictions": [], "embedding": [1.0, 0.0], "latency_seconds": 0.001}

    model.predict_one = synthetic_predict
    try:
        receipt = model.infer(output, splits=("dev",), crash_after=1 if phase == "interrupt" else None)
    except SystemExit as exc:
        if phase != "interrupt" or "Injected driver interruption" not in str(exc):
            raise
        print("Process stopped after second job was claimed, before its result commit.")
        return 73
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


def journal(output):
    dbs = sorted((output / "work/jobs").glob("*.sqlite"))
    if len(dbs) != 1:
        raise ValueError(f"Expected one journal, found {len(dbs)}")
    with sqlite3.connect(dbs[0]) as con:
        rows = con.execute("SELECT key,sample_id,status,attempts,result FROM jobs ORDER BY sample_id").fetchall()
    return [{"key": key, "sample_id": sid, "status": status, "attempts": attempts,
             "has_result": result is not None} for key, sid, status, attempts, result in rows]


def run_child(output, phase):
    process = subprocess.run([sys.executable, __file__, "--child", phase, "--output", str(output)],
                             cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    (output / f"{phase}.log").write_text(process.stdout + process.stderr)
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New directory; never reuse a frozen experiment")
    parser.add_argument("--child", choices=("interrupt", "resume", "replay"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.child:
        raise SystemExit(child(output, args.child))
    setup(output)
    first = run_child(output, "interrupt")
    before = journal(output)
    if first.returncode != 73 or [(r["sample_id"], r["status"], r["attempts"]) for r in before] != [
        ("a", "succeeded", 1), ("b", "running", 1)]:
        raise ValueError("Unexpected interrupted state; inspect interrupt.log")
    prediction_path = output / "reports/pilot/baseline/predictions.json"
    if prediction_path.exists():
        raise ValueError("Partial final predictions were published")

    second = run_child(output, "resume")
    if second.returncode:
        raise ValueError("Resume failed; inspect resume.log")
    receipt = json.loads(second.stdout)
    after = journal(output)
    if (receipt["recovered_jobs"], receipt["computed"], receipt["cache_hits"]) != (1, 2, 1):
        raise ValueError("Recovery receipt mismatch")
    if [(r["sample_id"], r["status"], r["attempts"], r["has_result"]) for r in after] != [
        ("a", "succeeded", 1, True), ("b", "succeeded", 2, True),
        ("c", "succeeded", 1, True)]:
        raise ValueError("Recovered journal mismatch")
    prediction = json.loads(prediction_path.read_text())
    if [r["sample_id"] for r in prediction["rows"]] != ["a", "b", "c"]:
        raise ValueError("Final prediction coverage/order mismatch")
    final_hash = file_hash(prediction_path)

    third = run_child(output, "replay")
    if third.returncode:
        raise ValueError("Replay failed; inspect replay.log")
    replay = json.loads(third.stdout)
    if (replay["recovered_jobs"], replay["computed"], replay["cache_hits"]) != (0, 0, 3):
        raise ValueError("Idempotent replay receipt mismatch")
    if journal(output) != after or file_hash(prediction_path) != final_hash:
        raise ValueError("Replay changed committed state or output")
    summary = {"scope": "synthetic computation; real infer orchestration, subprocess exit, SQLite and file writes",
               "interrupted_exit_code": first.returncode, "before": before,
               "resume": {k: receipt[k] for k in ("recovered_jobs", "computed", "cache_hits", "job_counts")},
               "after": after,
               "replay": {k: replay[k] for k in ("recovered_jobs", "computed", "cache_hits")},
               "final_predictions_sha256": final_hash,
               "limits": "Single runner on a local filesystem; at-least-once computation, one committed result per content key. No neural inference or production deployment."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
