"""Verify frozen release evidence and explicit presentation-only updates."""

from pathlib import Path

from driving_data.common import file_hash, read_json


def verify(root):
    manifest = root / "reports/artifact_manifest.json"
    receipt = read_json(manifest)
    expected = dict(receipt["files"])
    corrections_path = root / "configs/presentation_updates.json"
    if corrections_path.exists():
        correction = read_json(corrections_path)
        if correction["base_manifest_sha256"] != file_hash(manifest):
            raise ValueError("Release manifest identity changed")
        for relative, update in correction["updates"].items():
            path = Path(relative)
            # Presentation maintenance cannot exempt research inputs or results.
            allowed = relative in {
                "README.md",
                "dashboard.py",
                "docs/ARCHITECTURE.md",
                "docs/IMPLEMENTATION_MAP.md",
                "docs/RUNNING.md",
                "scripts/verify_artifacts.py",
                "tests/test_dashboard.py",
                "tests/test_artifact_updates.py",
            }
            if path.is_absolute() or ".." in path.parts or not allowed:
                raise ValueError(f"Not a presentation update: {relative}")
            if expected.get(relative) != update["previous_sha256"]:
                raise ValueError(f"Previous release identity mismatch: {relative}")
            if not update["reason"].strip():
                raise ValueError(f"Missing update reason: {relative}")
            expected[relative] = update["current_sha256"]
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Artifact mismatch: {relative}")
    return len(expected)


if __name__ == "__main__":
    count = verify(Path(__file__).resolve().parents[1])
    print(f"Verified {count} source/config/evidence files; frozen manifest preserved")
