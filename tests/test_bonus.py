"""
Additional tests for the Q&A Pipeline (bonus).

These use stubs instead of the real embedding model and LLM, so the whole
file runs in milliseconds. They cover the prompt assembly, the CLI loop,
and the error paths that the end-to-end tests in test_pipeline.py do not.

Run: pytest tests/test_bonus.py -v
"""

import builtins

import pytest

from src import pipeline
from src.pipeline import ask_question


# ────────────────────────────────
# Stubs
# ────────────────────────────────
class FakeDoc:
    """Minimal stand-in for a LangChain Document."""

    def __init__(self, page_content: str):
        self.page_content = page_content


class FakeVectorStore:
    """Records the query it was given and returns fixed chunks."""

    def __init__(self, chunks=("chunk one", "chunk two", "chunk three")):
        self.chunks = chunks
        self.last_query = None
        self.last_k = None

    def similarity_search(self, query, k=3):
        self.last_query = query
        self.last_k = k
        return [FakeDoc(c) for c in self.chunks[:k]]


class FakeLLM:
    """Records the prompt it was given and returns a fixed answer."""

    def __init__(self, answer="a stub answer"):
        self.answer = answer
        self.last_prompt = None

    def __call__(self, prompt):
        self.last_prompt = prompt
        return [{"generated_text": self.answer}]


@pytest.fixture
def store():
    return FakeVectorStore()


@pytest.fixture
def fake_llm():
    return FakeLLM()


# ────────────────────────────────
# Prompt assembly
# ────────────────────────────────
class TestPromptAssembly:
    def test_retrieves_three_chunks(self, store, fake_llm):
        ask_question(store, fake_llm, "anything")
        assert store.last_k == 3, "should request the top 3 chunks"

    def test_searches_with_the_question(self, store, fake_llm):
        ask_question(store, fake_llm, "how much is the Growth package?")
        assert store.last_query == "how much is the Growth package?"

    def test_prompt_contains_every_chunk(self, store, fake_llm):
        ask_question(store, fake_llm, "anything")
        for chunk in store.chunks:
            assert chunk in fake_llm.last_prompt, f"{chunk!r} missing from prompt"

    def test_prompt_contains_the_question(self, store, fake_llm):
        ask_question(store, fake_llm, "a very distinctive question")
        assert "a very distinctive question" in fake_llm.last_prompt

    def test_prompt_uses_the_template(self, store, fake_llm):
        ask_question(store, fake_llm, "anything")
        assert "Context:" in fake_llm.last_prompt
        assert "Client question:" in fake_llm.last_prompt

    def test_answer_comes_from_the_llm(self, store):
        result = ask_question(store, FakeLLM("forty two"), "anything")
        assert result["answer"] == "forty two"

    def test_sources_are_the_retrieved_chunks(self, store, fake_llm):
        result = ask_question(store, fake_llm, "anything")
        assert result["sources"] == list(store.chunks)

    def test_sources_are_all_strings(self, store, fake_llm):
        result = ask_question(store, fake_llm, "anything")
        assert all(isinstance(s, str) for s in result["sources"])


# ────────────────────────────────
# CLI: interactive loop and --query
# ────────────────────────────────
@pytest.fixture
def cli(monkeypatch, store, fake_llm):
    """Patch out model loading so main() runs instantly.

    Returns a helper that runs main() with the given argv and stdin lines,
    and reports which questions reached ask_question().
    """
    asked = []
    calls = {"input": 0}

    def fake_ask(vector_store, llm, question):
        asked.append(question)
        return {"answer": "stub", "sources": ["src"]}

    monkeypatch.setattr(pipeline, "build_knowledge_base", lambda d: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(pipeline, "ask_question", fake_ask)

    def run(argv=(), stdin=()):
        monkeypatch.setattr("sys.argv", ["pipeline", *argv])
        lines = iter(stdin)

        def fake_input(prompt=""):
            calls["input"] += 1
            try:
                return next(lines)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr(builtins, "input", fake_input)
        pipeline.main()
        return asked

    run.input_calls = calls
    return run


class TestInteractiveLoop:
    def test_quit_exits_cleanly(self, cli):
        """Regression: the loop must not raise on the way out.

        Nothing in test_pipeline.py imports main(), so a leftover
        NotImplementedError below the loop survived a fully green test run
        and crashed the CLI on every exit. This test covers that gap.
        """
        assert cli(stdin=["quit"]) == []

    def test_quit_is_case_and_space_insensitive(self, cli):
        assert cli(stdin=["  QUIT  "]) == []

    def test_question_reaches_ask_question(self, cli):
        assert cli(stdin=["what is SEO?", "quit"]) == ["what is SEO?"]

    def test_empty_input_is_skipped(self, cli):
        """Blank lines must not be sent to the retriever."""
        assert cli(stdin=["", "   ", "real question", "quit"]) == ["real question"]

    def test_eof_exits_cleanly(self, cli):
        """Ctrl+D must not raise EOFError out of main()."""
        assert cli(stdin=[]) == []

    def test_keyboard_interrupt_exits_cleanly(self, cli, monkeypatch):
        """Ctrl+C must not raise out of main()."""
        def interrupt(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr(builtins, "input", interrupt)
        cli(stdin=[])

    def test_interrupt_during_generation_exits_cleanly(self, cli, monkeypatch):
        """Ctrl+C while the model is answering must not escape main().

        Handling KeyboardInterrupt only around input() left the multi-second
        generation window unguarded — exactly when a user is most likely to
        press Ctrl+C. It exited with a traceback and code -2.
        """
        def interrupt(vector_store, llm, question):
            raise KeyboardInterrupt

        monkeypatch.setattr(pipeline, "ask_question", interrupt)
        cli(stdin=["a question", "quit"])


class TestQueryFlag:
    def test_query_answers_once(self, cli):
        assert cli(argv=["--query", "how much?"]) == ["how much?"]

    def test_query_does_not_enter_the_loop(self, cli):
        """--query must return before prompting; input() should never be called."""
        cli(argv=["--query", "how much?"])
        assert cli.input_calls["input"] == 0

    def test_empty_query_is_rejected(self, cli):
        with pytest.raises(SystemExit) as exc:
            cli(argv=["--query", "   "])
        assert "empty" in str(exc.value)


# ────────────────────────────────
# Missing / empty data directory
# ────────────────────────────────
class TestDataDirErrors:
    def test_missing_directory_reports_clearly(self, monkeypatch):
        def boom(data_dir):
            raise FileNotFoundError(f"Directory not found: '{data_dir}'")

        monkeypatch.setattr(pipeline, "build_knowledge_base", boom)
        with pytest.raises(SystemExit) as exc:
            pipeline.load_pipeline("data")
        assert "data directory not found" in str(exc.value)

    def test_empty_directory_reports_clearly(self, monkeypatch):
        """FAISS raises a bare IndexError on zero documents; translate it."""
        def boom(data_dir):
            raise IndexError("list index out of range")

        monkeypatch.setattr(pipeline, "build_knowledge_base", boom)
        with pytest.raises(SystemExit) as exc:
            pipeline.load_pipeline("data")
        assert "no .txt documents found" in str(exc.value)
