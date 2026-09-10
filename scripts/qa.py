"""Write an actual local QA receipt and a hash manifest for distributable evidence."""

import platform
import subprocess
import sys
from pathlib import Path

from driving_data.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
checks = []
for command in [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts", "dashboard.py"],
]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    checks.append(
        {
            "command": "python " + " ".join(command[1:]),
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    )
    if result.returncode:
        print(result.stdout, result.stderr)
        raise SystemExit(result.returncode)
repro = read_json(ROOT / "reports/pilot/reproducibility.json")
assert repro["status"] == "passed"
write_json(
    ROOT / "reports/qa.json",
    {
        "status": "passed",
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "checks": checks,
        "reproducibility": "reports/pilot/reproducibility.json",
        "remote_ci": "See GitHub Actions for per-commit remote status; this is a local receipt",
        "publication": "Publication is verified separately from local QA",
    },
)
files = []
for folder in ("src", "configs", "tests", "scripts", "docs", "assets", "reports", ".github", ".streamlit"):
    files.extend(
        p
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and "local" not in p.relative_to(ROOT).parts
        and not any(part.endswith(".egg-info") for part in p.parts)
    )
files.extend(
    ROOT / p
    for p in (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "requirements-lock.txt",
        "dashboard.py",
        ".gitignore",
        ".gitattributes",
    )
)
files = [p for p in files if p != ROOT / "reports/artifact_manifest.json"]
write_json(
    ROOT / "reports/artifact_manifest.json",
    {"algorithm": "sha256", "files": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}},
)
print(f"QA passed; manifest contains {len(files)} source/config/evidence files")
