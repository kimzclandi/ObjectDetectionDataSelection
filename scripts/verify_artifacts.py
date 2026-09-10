"""Check committed evidence integrity without image downloads or model weights."""

from pathlib import Path

from driving_data.common import file_hash, read_json

root = Path(__file__).resolve().parents[1]
receipt = read_json(root / "reports/artifact_manifest.json")
for relative, expected in receipt["files"].items():
    path = root / relative
    if not path.is_file() or file_hash(path) != expected:
        raise SystemExit(f"Artifact mismatch: {relative}")
print(f"Verified {len(receipt['files'])} source/config/evidence files")
