# Full Stack Intern Take-Home Assignment

## 🎯 Objective

Build an **interactive Q&A chatbot** for a marketing agency using LangChain and a local LLM. A potential client can ask questions about services, pricing, and process — and get answers pulled from the agency's docs.

No API keys. No GPU. Everything runs locally.

---

## ⏱️ Time Expectation

~1-2 days

## 📋 What You'll Build

An interactive CLI where a client asks questions and gets answers:

```
> How much does the Growth package cost?

📄 Sources:
  1. GROWTH PACKAGE — $5,500/month. Best for scaling businesses...

💬 Answer: The Growth package costs $5,500 per month.

> Can I cancel early?

📄 Sources:
  1. Early termination before the minimum commitment requires...

💬 Answer: Yes, with 50% of the remaining contract value.
```

---

## 🧰 Stack

| Component    | Library / Tool                          |
| ------------ | --------------------------------------- |
| Framework    | LangChain (v0.3.x)                      |
| Embeddings   | HuggingFace (`all-MiniLM-L6-v2`, local) |
| Vector Store | FAISS (local)                           |
| LLM          | `google/flan-t5-base` (local, CPU)      |
| Testing      | pytest                                  |

---

## 🚀 Getting Started

1. **Fork** this repo to your own GitHub account
2. Clone your fork and `cd` into it
3. Set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the tests (they will fail until you complete the TODOs):

```bash
pytest tests/ -v
```

> First run downloads two models (~1.2GB total). Cached after that.

---

## 📝 Your Tasks

Open `src/pipeline.py` — there are **2 TODOs**.

The document loading, chunking, embeddings, and vector store are **already built** in `knowledge_base.py`. Don't modify that file. You're building the response layer.

### TODO 1 — Implement `ask_question()`

Write a function that:

1. Searches the vector store for the 3 most relevant chunks
2. Combines their text into a context string
3. Plugs it into the provided prompt template
4. Calls the LLM and returns the answer + sources

**Hint — searching the vector store:**

```python
docs = vector_store.similarity_search("some query", k=3)
text = docs[0].page_content  # the actual text
```

**Hint — calling the LLM:**

```python
result = llm("some prompt")
answer = result[0]["generated_text"]
```

### TODO 2 — Complete the `main()` interactive loop

Write a loop that:

1. Builds the knowledge base and loads the LLM (helpers provided)
2. Takes user input
3. Calls `ask_question()` and prints the result
4. Exits on `quit`

---

## ✅ Evaluation

```bash
pytest tests/ -v
```

| Criteria                       | Weight |
| ------------------------------ | ------ |
| All tests pass                 | 40%    |
| Code clarity and structure     | 25%    |
| Correct retrieval + generation | 25%    |
| Bonus (see below)              | 10%    |

### Bonus (optional)

- Error handling (empty input, missing files)
- `--query` CLI argument for single-question mode
- Additional test cases
- Type hints

---

## 📁 Project Structure

```
langchain-intern-assignment/
├── README.md
├── requirements.txt
├── data/
│   ├── services.txt              ← agency service descriptions
│   ├── pricing.txt               ← packages and pricing
│   └── faq.txt                   ← client FAQ and process
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py         ← PRE-BUILT (do not modify)
│   └── pipeline.py               ← YOUR WORK GOES HERE
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## ⚠️ Troubleshooting

**`command not found: python`** — Use `python3`.

**`ModuleNotFoundError`** — Activate venv and run `pip install -r requirements.txt`.

**Slow first run** — Models download once (~1.2GB), then cached.

---

## ❓ FAQ

**Do I need an API key?** No.

**What Python version?** 3.10+

**Can I modify `knowledge_base.py`?** No.

---

Good luck! 🚀

---

# Implementation Notes

_Everything below was added by me. Sections above are the original assignment._

## What was built

| Item                        | Where                            |
| --------------------------- | -------------------------------- |
| TODO 1 — `ask_question()`   | `src/pipeline.py`                |
| TODO 2 — interactive loop   | `src/pipeline.py`                |
| Bonus — error handling      | `main()` / `load_pipeline()`     |
| Bonus — `--query` flag      | `main()`                         |
| Bonus — additional tests    | `tests/test_bonus.py` (20 tests) |
| Bonus — type hints          | throughout                       |

`src/knowledge_base.py` and `tests/test_pipeline.py` were not modified.

A missing or empty `data/` directory exits with a clear message rather than an
unhandled traceback, and blank input, `Ctrl+C`, and `Ctrl+D` are all handled in
the interactive loop.

## Running it

```bash
python -m src.pipeline                                             # interactive
python -m src.pipeline --query "How much is the Growth package?"   # single question
pytest tests/ -v                                                   # 30 tests
pytest tests/test_bonus.py -v                                      # fast subset, ~3s
```

Run as `-m src.pipeline`, not `python src/pipeline.py` — the module uses an
absolute import (`from src.knowledge_base import ...`), so the repo root needs
to be on the path.

Type hints check clean with:

```bash
mypy --ignore-missing-imports src tests
```

## Testing approach

The original end-to-end tests validate retrieval and generation using the real
local models.

I added `tests/test_bonus.py` with stubbed `FakeVectorStore` and `FakeLLM`
implementations so control-flow and prompt-assembly tests run without
downloading or loading models. The file completes in ~3 seconds.

The additional 20 tests cover:

- Prompt assembly and retrieval (`k=3`)
- CLI behaviour and `--query`
- Empty input
- `quit`, `Ctrl+D`, and `Ctrl+C`
- Interrupts during generation
- Missing and empty data directories

## Known limitations

### Local model response style

The assignment uses `google/flan-t5-base` on CPU. The model sometimes produces
shorter answers than the examples in the assignment, particularly for yes/no
questions.

For example, the retrieved source correctly contains:

> Early termination before the minimum commitment requires payment of 50% of
> the remaining contract value.

but the model answers:

> Yes.

This is a generation-model limitation rather than a retrieval failure — feeding
the model only the single correct chunk, and prompting it explicitly for
amounts and conditions, both still produce `Yes.` The retrieved sources remain
available to the caller and are printed with every answer.

I kept the specified `flan-t5-base` model and the provided `get_llm()`
unchanged rather than substituting a larger model, since the assignment
requires the provided local CPU-based stack.

### Document chunking

The provided `knowledge_base.py` uses a fixed 500-character chunk size. Some
pricing entries are split into a header-only chunk and a body chunk, which can
occasionally cause duplicate or incomplete retrieval results.

I left `knowledge_base.py` unchanged because the assignment explicitly requires
it to remain untouched and specifies `k=3` retrieval.

### Corpus scope

The provided `data/` directory contains both agency documents and unrelated
Acme Corp documents. Because the assignment's loader indexes every `.txt` file
and the task does not specify document-level filtering, retrieval can
occasionally return unrelated content for off-domain questions.

I left this behaviour unchanged rather than modifying the provided knowledge
base.
