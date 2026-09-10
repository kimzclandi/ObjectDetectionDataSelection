from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_fresh_test_dashboard_shows_actual_negative_result():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard.py"), default_timeout=10
    ).run()
    app.sidebar.selectbox[0].select("失败驱动训练 · 新评测集").run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["8.595", "9.229", "7.609"]
    next(w for w in app.selectbox if w.label == "采样运行").select("random-s211").run()
    assert not app.exception
