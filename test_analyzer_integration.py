import importlib


def test_analyze_content_integration(monkeypatch):
    """Verify that the `analyze_content` node sets `analysis` on AgentState.

    This test monkeypatches the Analyzer used by `app.analyze_content`
    with a lightweight dummy to avoid optional dependencies.
    """
    app = importlib.import_module('app')

    class DummyAnalyzer:
        def analyze(self, text):
            return {"dummy": True, "word_count": len(text.split())}

    # Replace the Analyzer in the app module with our dummy
    monkeypatch.setattr(app, 'Analyzer', DummyAnalyzer)

    state = app.AgentState()
    state.generated_text = "one two three"

    new_state = app.analyze_content(state)

    assert isinstance(new_state.analysis, dict)
    assert new_state.analysis.get('dummy') is True
    assert new_state.analysis.get('word_count') == 3
