from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_saved_evidence_viewer_and_filter():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "dashboard.py")).run(timeout=30)
    assert not app.exception
    if (root / "reports/pilot/experiment.json").exists():
        assert [m.value for m in app.metric[:4]] == ["240", "101", "59", "9"]
        app.selectbox[1].select("night").run()
        assert not app.exception
        app.selectbox[0].select("diverse-s17").run()
        assert not app.exception
