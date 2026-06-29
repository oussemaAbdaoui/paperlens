"""PaperLens — Streamlit UI (v1 frontend, per spec §7).

Thin client over the FastAPI backend. Three tabs:
  * Corpus  — upload PDFs, import by arXiv/DOI, manage collections, edit metadata.
  * Search  — semantic search with metadata filters (M6/M7).
  * Ask     — grounded, cited QA (M8) with clickable source chunks.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("PAPERLENS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PaperLens", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(path: str, **kw):
    return requests.get(f"{API}{path}", timeout=30, **kw)


def api_post(path: str, **kw):
    return requests.post(f"{API}{path}", timeout=120, **kw)


def load_collections() -> list[dict]:
    try:
        r = api_get("/collections")
        return r.json() if r.ok else []
    except requests.RequestException:
        return []


def collection_picker(label: str, *, allow_all: bool):
    cols = load_collections()
    options = {c["name"]: c["id"] for c in cols}
    labels = (["All collections"] if allow_all else ["(Uncategorized)"]) + list(options)
    chosen = st.selectbox(label, labels)
    if chosen in ("All collections", "(Uncategorized)"):
        return None
    return options[chosen]


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔬 PaperLens")
st.caption("AI-powered research assistant — grounded, cited answers across your paper corpus.")

try:
    health = api_get("/health").json()
    st.sidebar.success(f"API online · {health['indexed_vectors']} chunks indexed")
    st.sidebar.write(f"**LLM:** {health['llm_model']}")
    st.sidebar.write(f"**Embeddings:** {health['embedding_backend']}")
except Exception:
    st.sidebar.error(f"Cannot reach API at {API}")

tab_corpus, tab_search, tab_ask = st.tabs(["📚 Corpus", "🔎 Search", "💬 Ask"])


# --------------------------------------------------------------------------- #
# Corpus tab (M1 / M2)
# --------------------------------------------------------------------------- #
with tab_corpus:
    left, right = st.columns(2)

    with left:
        st.subheader("Collections")
        with st.form("new_collection", clear_on_submit=True):
            name = st.text_input("New collection name")
            if st.form_submit_button("Create") and name.strip():
                r = api_post("/collections", json={"name": name.strip()})
                st.success("Created") if r.ok else st.error(r.text)
        for c in load_collections():
            st.write(f"- **{c['name']}** · {c['paper_count']} papers")

    with right:
        st.subheader("Add papers")
        coll_id = collection_picker("Add to collection", allow_all=False)

        up = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if up and st.button("Upload"):
            for f in up:
                data = {"on_duplicate": "skip"}
                if coll_id is not None:
                    data["collection_id"] = coll_id
                r = api_post("/papers/upload", files={"file": (f.name, f.getvalue(), "application/pdf")}, data=data)
                if r.ok:
                    st.success(f"{f.name}: {r.json().get('message')}")
                else:
                    st.error(f"{f.name}: {r.text}")

        st.markdown("**Import by arXiv ID or DOI**")
        ident = st.text_input("e.g. 2301.07041 or 10.1145/3292500", key="import_id")
        if st.button("Import") and ident.strip():
            payload = {"identifier": ident.strip()}
            if coll_id is not None:
                payload["collection_id"] = coll_id
            r = api_post("/papers/import", json=payload)
            st.success(r.json().get("message")) if r.ok else st.error(r.text)

    st.divider()
    st.subheader("Papers")
    papers = api_get("/papers").json() if api_get("/papers").ok else []
    for p in papers:
        meta = p["metadata"]
        status_icon = {"Indexed": "✅", "Processing": "⏳", "Failed": "❌"}.get(p["status"], "•")
        with st.expander(f"{status_icon} {meta.get('title') or p['filename']} ({p['status']})"):
            st.write(f"**Authors:** {', '.join(meta.get('authors') or []) or '—'}")
            st.write(f"**Year:** {meta.get('year') or '—'} · **Venue:** {meta.get('venue') or '—'}")
            st.write(f"**DOI:** {meta.get('doi') or '—'} · **Lang:** {meta.get('language') or '—'}")
            st.write(f"**Pages:** {p.get('num_pages') or '—'} · **Chunks:** {p.get('num_chunks')}")
            if p["status"] == "Failed" and p.get("error"):
                st.error(p["error"])
                if st.button("Retry indexing", key=f"retry_{p['id']}"):
                    api_post(f"/papers/{p['id']}/reindex")
                    st.rerun()
            if st.button("Delete", key=f"del_{p['id']}"):
                requests.delete(f"{API}/papers/{p['id']}", timeout=30)
                st.rerun()


# --------------------------------------------------------------------------- #
# Search tab (M6 / M7)
# --------------------------------------------------------------------------- #
def filter_inputs(key: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        cid = collection_picker("Collection", allow_all=True)
    with c2:
        ymin = st.number_input("Year ≥", min_value=0, max_value=2100, value=0, key=f"ymin_{key}")
    with c3:
        ymax = st.number_input("Year ≤", min_value=0, max_value=2100, value=0, key=f"ymax_{key}")
    lang = st.text_input("Language (ISO code, optional)", key=f"lang_{key}")
    filters: dict = {}
    if cid is not None:
        filters["collection_id"] = cid
    if ymin:
        filters["year_min"] = int(ymin)
    if ymax:
        filters["year_max"] = int(ymax)
    if lang.strip():
        filters["language"] = lang.strip()
    return filters


with tab_search:
    st.subheader("Semantic search")
    q = st.text_input("Query", key="search_q")
    filters = filter_inputs("search")
    if st.button("Search") and q.strip():
        r = api_post("/search", json={"query": q.strip(), "filters": filters})
        if not r.ok:
            st.error(r.text)
        else:
            data = r.json()
            st.caption(f"{len(data['results'])} results · {data['latency_ms']:.0f} ms")
            for res in data["results"]:
                st.markdown(
                    f"**{res['title'] or 'Untitled'}** "
                    f"({', '.join(res['authors']) or '—'}, {res['year'] or 'n.d.'}) — "
                    f"*{res['section']}*, p. {res['page']} · score {res['score']:.3f}"
                )
                st.write(res["text"])
                st.divider()


# --------------------------------------------------------------------------- #
# Ask tab (M8)
# --------------------------------------------------------------------------- #
with tab_ask:
    st.subheader("Ask your corpus")
    question = st.text_area("Question", key="qa_q")
    filters = filter_inputs("qa")
    if st.button("Ask") and question.strip():
        with st.spinner("Retrieving and generating…"):
            r = api_post("/qa", json={"question": question.strip(), "filters": filters})
        if not r.ok:
            st.error(r.text)
        else:
            data = r.json()
            if not data["grounded"]:
                st.warning(data["answer"])
            else:
                st.markdown(data["answer"])
                st.caption(f"Answered in {data['latency_ms']:.0f} ms retrieval")
                st.markdown("#### Sources")
                for src in data["sources"]:
                    with st.expander(
                        f"{src['title'] or 'Untitled'} — {src['section']}, p. {src['page']}"
                    ):
                        st.write(src["text"])
