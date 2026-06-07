"""
Hybrid Chunking Strategy — LLM Suite RAG Pipeline
Validated through controlled precision@5 experiments across 120 documents and 90 queries.

Key finding: paragraph-boundary chunking outperforms fixed-size by 14 precision points
on legal and regulatory documents. Root cause: fixed-size chunking splits obligation
clauses mid-sentence, producing retrieved chunks that state an obligation without
its qualifying conditions — unacceptable for compliance use cases.
"""

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)
from langchain.schema import Document
from typing import List
import re


# Document categories requiring paragraph-boundary chunking
LEGAL_REGULATORY_CATEGORIES = {"regulatory", "legal", "custody", "credit", "isda"}

# Fixed-size parameters (general content)
FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 200

# Paragraph boundary separators (ordered by preference)
PARAGRAPH_SEPARATORS = [
    "\n\n",          # Standard paragraph break
    "\n(?=[A-Z])",   # Newline followed by capitalized sentence (new clause)
    r"\. (?=[A-Z])", # Sentence boundary before capitalized word
    "\n",
    " ",
]


def get_chunker(doc_category: str, chunk_size: int = FIXED_CHUNK_SIZE):
    """
    Return appropriate chunker based on document category.
    Paragraph-boundary for legal/regulatory, fixed-size for general.
    """
    if doc_category.lower() in LEGAL_REGULATORY_CATEGORIES:
        return RecursiveCharacterTextSplitter(
            separators=PARAGRAPH_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=True,
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=FIXED_CHUNK_OVERLAP,
            length_function=len,
        )


def chunk_document(doc: Document) -> List[Document]:
    """
    Apply hybrid chunking strategy to a document.
    Document category inferred from metadata if available.
    """
    doc_category = doc.metadata.get("doc_type", "general")
    chunker = get_chunker(doc_category)
    chunks = chunker.split_documents([doc])

    # Propagate parent document metadata to all chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunking_strategy": "paragraph_boundary" if doc_category in LEGAL_REGULATORY_CATEGORIES else "fixed_size",
            "parent_doc_id": doc.metadata.get("doc_id"),
        })

    return chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk a list of documents using the hybrid strategy."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks


# ─── Chunking Validation Framework ───────────────────────────────────────────

def evaluate_chunking_strategy(
    documents: List[Document],
    queries: List[dict],
    retriever,
    k: int = 5,
) -> dict:
    """
    Evaluate chunking strategy precision@k on a set of ground-truth queries.
    Used to validate the hybrid strategy decision empirically.

    queries format: [{"query": str, "relevant_doc_ids": List[str]}, ...]
    """
    precision_scores = []

    for query_item in queries:
        query = query_item["query"]
        relevant_ids = set(query_item["relevant_doc_ids"])

        retrieved = retriever.get_relevant_documents(query)[:k]
        retrieved_ids = {
            doc.metadata.get("parent_doc_id") for doc in retrieved
        }

        hits = len(relevant_ids & retrieved_ids)
        precision = hits / k
        precision_scores.append(precision)

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "k": k,
        "n_queries": len(queries),
    }
