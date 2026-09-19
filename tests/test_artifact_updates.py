"""Presentation corrections cannot mask research-evidence changes."""

import importlib.util
import json
from pathlib import Path

import pytest

from driving_data.common import file_hash

spec = importlib.util.spec_from_file_location(
    "verify_artifacts", Path(__file__).parents[1] / "scripts/verify_artifacts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "relative",
    [
        "reports/result.json",
        "data/manifest.json",
        "src/model.py",
        "docs/FAILURE_V2_PROTOCOL.md",
        "docs/FAILURE_V2_REPORT.md",
        "scripts/diagnose_failure_v3.py",
        "tests/test_failure_v2.py",
        "../README.md",
    ],
)
def test_research_evidence_cannot_be_exempted(tmp_path, relative):
    (tmp_path / "reports").mkdir()
    (tmp_path / "configs").mkdir()
    manifest = tmp_path / "reports/artifact_manifest.json"
    manifest.write_text(json.dumps({"files": {relative: "old"}}))
    correction = {
        "base_manifest_sha256": file_hash(manifest),
        "updates": {relative: {"previous_sha256": "old", "current_sha256": "new", "reason": "invalid"}},
    }
    (tmp_path / "configs/presentation_updates.json").write_text(json.dumps(correction))
    with pytest.raises(ValueError, match="Not a presentation update"):
        module.verify(tmp_path)


def test_presentation_update_still_checks_old_identity_and_new_bytes(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "configs").mkdir()
    document = tmp_path / "README.md"
    document.write_text("old")
    old = file_hash(document)
    manifest = tmp_path / "reports/artifact_manifest.json"
    manifest.write_text(json.dumps({"files": {"README.md": old}}))
    document.write_text("new")
    correction = {
        "base_manifest_sha256": file_hash(manifest),
        "updates": {
            "README.md": {
                "previous_sha256": old,
                "current_sha256": file_hash(document),
                "reason": "navigation",
            }
        },
    }
    path = tmp_path / "configs/presentation_updates.json"
    path.write_text(json.dumps(correction))
    assert module.verify(tmp_path) == 1
    document.write_text("tampered")
    with pytest.raises(ValueError, match="Artifact mismatch"):
        module.verify(tmp_path)
    correction["updates"]["README.md"]["previous_sha256"] = "wrong"
    path.write_text(json.dumps(correction))
    with pytest.raises(ValueError, match="Previous release identity mismatch"):
        module.verify(tmp_path)
