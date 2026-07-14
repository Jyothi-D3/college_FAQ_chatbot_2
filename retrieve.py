"""
Phase 2 -- Retrieval
=====================
Uses local HuggingFace embeddings (no API key needed) for vector search.

Usage:
    python retrieve.py
    python retrieve.py --query "What is the hostel fee?" --top-k 5
    python retrieve.py --list-sections

Environment:
    GROQ_API_KEY  -- required for generation (not needed for retrieval alone)
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "college_faq"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local HuggingFace model, no API needed

SMOKE_TEST_QUERIES = [
    "What is the tuition fee for CSE?",
    "What is the highest placement package?",
    "Is BVRIT Hyderabad accredited by NAAC?",
]


def load_store(persist_dir: str, collection_name: str, embedding_model: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    if not os.path.isdir(persist_dir):
        print(f"No index found at '{persist_dir}'. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    count = store._collection.count()
    if count == 0:
        print(f"Collection '{collection_name}' at '{persist_dir}' is empty. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)
    return store


def list_sections(store) -> list[str]:
    raw = store._collection.get(include=["metadatas"])
    sections = sorted({m.get("section", "") for m in raw["metadatas"] if m.get("section")})
    return sections


def retrieve(store, query: str, top_k: int = 5, section: str | None = None):
    search_kwargs = {"k": top_k}
    if section:
        search_kwargs["filter"] = {"section": section}
    retriever = store.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def print_results(query: str, docs, section_filter: str | None):
    scope = f" (scoped to section: '{section_filter}')" if section_filter else ""
    print(f"\nQuery: {query}{scope}")
    print("-" * 70)
    if not docs:
        print("  (no chunks retrieved)")
        return
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        loc = meta.get("section", "?")
        if meta.get("subsection"):
            loc += f" > {meta['subsection']}"
        print(f"  [{i}] section: {loc}")
        preview = doc.page_content.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"      {preview}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    ap.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--query", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--section", default=None)
    ap.add_argument("--list-sections", action="store_true")
    args = ap.parse_args()

    store = load_store(args.persist_dir, args.collection_name, args.embedding_model)

    if args.list_sections:
        print("Sections available:")
        for s in list_sections(store):
            print(f"  - {s}")
        return

    if args.query:
        docs = retrieve(store, args.query, top_k=args.top_k, section=args.section)
        print_results(args.query, docs, args.section)
        return

    print(f"Running {len(SMOKE_TEST_QUERIES)} smoke-test queries ...")
    for q in SMOKE_TEST_QUERIES:
        docs = retrieve(store, q, top_k=args.top_k, section=args.section)
        print_results(q, docs, args.section)


if __name__ == "__main__":
    main()
