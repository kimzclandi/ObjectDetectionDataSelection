from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_saved_evidence_viewer_and_filter():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "dashboard.py")).run(timeout=30)
    assert not app.exception
    assert [m.value for m in app.metric] == ["8.595", "9.229", "7.609"]
    app.sidebar.selectbox[0].select("历史训练 / VLM 推理").run()
    if (root / "reports/pilot/experiment.json").exists():
        assert [m.value for m in app.metric[:4]] == ["240", "101", "59", "9"]
        next(w for w in app.selectbox if w.label == "光照切片").select("night").run()
        assert not app.exception
        next(w for w in app.selectbox if w.label == "模型运行").select("diverse-s17").run()
        assert not app.exception
