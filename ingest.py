"""
Phase 1 -- Ingest and index
============================
Uses local HuggingFace embeddings (no API key needed).

Usage:
    python ingest.py --docx "BVRIT_Hyderabad_Knowledge_Base.docx"

Environment:
    None required (uses local embeddings)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


# --------------------------------------------------------------------------
# 1. Parse the .docx in document order (paragraphs AND tables)
# --------------------------------------------------------------------------

@dataclass
class SectionBlock:
    section: str
    subsection: str | None
    text: str


def iter_block_items(document: DocxDocument):
    parent_elm = document.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def table_to_text(table: Table) -> str:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    header, *body = rows
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


def parse_docx_into_blocks(path: str) -> list[SectionBlock]:
    doc = DocxDocument(path)
    blocks: list[SectionBlock] = []

    current_section = "Untitled"
    current_subsection: str | None = None
    buffer: list[str] = []

    def flush():
        text = "\n".join(t for t in buffer if t.strip())
        if text.strip():
            blocks.append(SectionBlock(current_section, current_subsection, text))
        buffer.clear()

    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            style_obj = item.style
            style = (style_obj.name or "").lower() if style_obj is not None else ""
            text = item.text.strip()
            if not text:
                continue
            if style.startswith("heading 1") or style == "title":
                flush()
                current_section = text
                current_subsection = None
            elif style.startswith("heading 2"):
                flush()
                current_subsection = text
            else:
                prefix = "- " if style.startswith("list") else ""
                buffer.append(prefix + text)
        elif isinstance(item, Table):
            table_text = table_to_text(item)
            if table_text:
                buffer.append(table_text)

    flush()
    return [b for b in blocks if b.section != "Untitled"]


# --------------------------------------------------------------------------
# 2. Split blocks into chunks with metadata
# --------------------------------------------------------------------------

def chunk_blocks(blocks: list[SectionBlock], chunk_size: int, chunk_overlap: int, source_name: str):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for block in blocks:
        pieces = splitter.split_text(block.text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "metadata": {
                    "source": source_name,
                    "section": block.section,
                    "subsection": block.subsection or "",
                    "chunk_index": i,
                },
            })
    return chunks


# --------------------------------------------------------------------------
# 3. Embed + persist to ChromaDB
# --------------------------------------------------------------------------

def build_index(chunks, persist_dir: str, collection_name: str, embedding_model: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document as LCDocument

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    lc_docs = [LCDocument(page_content=c["text"], metadata=c["metadata"]) for c in chunks]

    store = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return store


def verify_persistence(persist_dir: str, collection_name: str, embedding_model: str, expected_count: int):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    reloaded = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    actual_count = reloaded._collection.count()
    print(f"\n[verify] Reloaded '{collection_name}' from '{persist_dir}'")
    print(f"[verify] Expected: {expected_count} | Actual: {actual_count} "
          f"| {'MATCH ✅' if actual_count == expected_count else 'MISMATCH ❌'}")
    return actual_count == expected_count


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 1: ingest and index the college knowledge-base doc.")
    ap.add_argument("--docx", required=True)
    ap.add_argument("--persist-dir", default="./chroma_db")
    ap.add_argument("--collection-name", default="college_faq")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.docx):
        print(f"File not found: {args.docx}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Parsing {args.docx} ...")
    blocks = parse_docx_into_blocks(args.docx)
    print(f"      -> {len(blocks)} section/subsection blocks found")
    for s in sorted(set(b.section for b in blocks)):
        print(f"         - {s}")

    if args.dry_run:
        print("\n[dry-run] Skipping embedding.")
        return

    print(f"\n[2/4] Chunking (size={args.chunk_size}, overlap={args.chunk_overlap}) ...")
    source_name = os.path.basename(args.docx)
    chunks = chunk_blocks(blocks, args.chunk_size, args.chunk_overlap, source_name)
    print(f"      -> {len(chunks)} total chunks")
    by_section = {}
    for c in chunks:
        by_section.setdefault(c["metadata"]["section"], 0)
        by_section[c["metadata"]["section"]] += 1
    for s, n in by_section.items():
        print(f"         - {s}: {n} chunks")

    print(f"\n[3/4] Embedding with '{args.embedding_model}' -> '{args.persist_dir}' ...")
    build_index(chunks, args.persist_dir, args.collection_name, args.embedding_model)
    print(f"      -> indexed {len(chunks)} chunks into '{args.collection_name}'")

    print("\n[4/4] Verifying persistence ...")
    verify_persistence(args.persist_dir, args.collection_name, args.embedding_model, expected_count=len(chunks))


if __name__ == "__main__":
    main()
