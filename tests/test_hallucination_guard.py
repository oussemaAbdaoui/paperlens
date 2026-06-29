"""MVP acceptance checklist (§9): 20 out-of-scope queries must all return the
'no relevant passages found' refusal with zero exceptions.

We index a tiny, tightly-scoped corpus, then fire 20 queries about unrelated
topics under a strict collection filter that matches nothing, guaranteeing zero
retrieved chunks — the hallucination guard must refuse every time.
"""

from __future__ import annotations

from app.models import SearchFilters
from app.services import ingest, qa
from app.services.retrieval import retrieve
from factories import make_doc, long_body

OUT_OF_SCOPE_QUERIES = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 World Cup?",
    "What is the boiling point of water?",
    "Explain quantum chromodynamics.",
    "What year did the Roman Empire fall?",
    "How does photosynthesis work?",
    "What is the population of Tokyo?",
    "Recommend a good pizza recipe.",
    "What is the airspeed velocity of a swallow?",
    "How tall is Mount Everest?",
    "What is the chemical formula for table salt?",
    "Who painted the Mona Lisa?",
    "How do I change a car tire?",
    "What is the GDP of Germany?",
    "Explain the rules of cricket.",
    "What is the speed of light?",
    "How many moons does Jupiter have?",
    "What is the best programming language?",
    "When is the next solar eclipse?",
]


def test_twenty_out_of_scope_queries_all_refuse(db, store, monkeypatch):
    # Index one paper into collection 1.
    cid = db.create_collection("NLP")
    doc = make_doc({"Introduction": long_body(20)})
    pid = db.insert_paper(filename="p.pdf", path="/fake/p.pdf", collection_id=cid)
    monkeypatch.setattr(ingest, "parse_pdf", lambda path: doc)
    ingest.index_paper(pid, db, store, persist=False)

    # Any LLM call would be a failure: the guard must fire before generation.
    monkeypatch.setattr(qa, "_call_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called!")))

    # Filter to a non-existent collection so retrieval returns nothing.
    empty_filter = SearchFilters(collection_id=99999)
    failures = 0
    for query in OUT_OF_SCOPE_QUERIES:
        chunks = retrieve(query, empty_filter, db, store)
        resp = qa.answer_question(query, chunks)
        if resp.grounded or resp.answer != qa.NO_PASSAGES_MESSAGE:
            failures += 1
    assert failures == 0, f"{failures}/20 queries failed the hallucination guard"
