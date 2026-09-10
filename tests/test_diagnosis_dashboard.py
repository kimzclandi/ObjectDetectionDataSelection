from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_diagnostic_dashboard_uses_saved_evidence():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "diagnosis_dashboard.py")).run()
    assert not app.exception
    assert app.metric[0].value == "0.988"
    assert len(app.dataframe) == 2
