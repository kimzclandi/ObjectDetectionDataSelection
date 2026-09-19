import pytest

from driving_data.cli import main


@pytest.mark.parametrize(
    "command", ["prepare", "infer", "select", "experiment", "benchmark", "controls", "vlm-quality", "catalog"]
)
def test_mutating_cli_refuses_frozen_release_before_execution(tmp_path, monkeypatch, command):
    report = tmp_path / "reports/artifact_manifest.json"
    report.parent.mkdir()
    report.write_text("frozen sentinel")
    monkeypatch.setattr("sys.argv", ["driving-data", "--root", str(tmp_path), command])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert report.read_text() == "frozen sentinel"
    assert list(tmp_path.iterdir()) == [report.parent]
