import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Road data pilot; run from repository root")
    p.add_argument("--root", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--limit", type=int, default=240)
    prep.add_argument("--workers", type=int, default=6)
    sub.add_parser("verify")
    inf = sub.add_parser("infer")
    inf.add_argument("--name", default="baseline")
    inf.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    inf.add_argument("--checkpoint", type=Path)
    inf.add_argument("--crash-after", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", default="baseline")
    sel = sub.add_parser("select")
    sel.add_argument("--budget", type=int, default=12)
    sub.add_parser("experiment")
    sub.add_parser("benchmark")
    sub.add_parser("catalog")
    sub.add_parser("controls")
    vlm = sub.add_parser("vlm-quality")
    vlm.add_argument("--constrained", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.root, args.limit, args.workers)
    elif args.command == "verify":
        from .data import verify

        result = verify(args.root)
    elif args.command == "infer":
        from .model import infer

        result = infer(args.root, args.name, args.checkpoint, args.device, crash_after=args.crash_after)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(args.root, args.name)
    elif args.command == "select":
        from .selection import run_selection

        result = run_selection(args.root, args.budget)
    elif args.command == "experiment":
        from .training import experiment

        result = experiment(args.root)
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(args.root)
    elif args.command == "controls":
        from .controls import controls

        result = controls(args.root)
    elif args.command == "vlm-quality":
        from .vlm_quality import vlm_quality

        result = vlm_quality(args.root, constrained=args.constrained)
    else:
        import duckdb

        from .common import write_json

        with duckdb.connect() as con:
            rows = con.execute(
                "SELECT split,timeofday,COUNT(*) AS images,AVG(mean_luminance) AS mean_luminance "
                "FROM read_parquet(?) GROUP BY split,timeofday ORDER BY split,timeofday",
                [str(args.root / "data/catalog.parquet")],
            ).df()
        result = {"engine": "DuckDB", "version": duckdb.__version__, "rows": rows.to_dict("records")}
        write_json(args.root / "reports/pilot/catalog_query.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
