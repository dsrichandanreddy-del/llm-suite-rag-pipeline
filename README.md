# LLM Suite RAG Pipeline — JPMorgan GenAI Platform

**Employer:** JPMorgan Chase | AI/ML & Data Analytics  
**Role:** Applied AI/ML Engineer  
**Timeline:** September 2024 – March 2025  
**Domain:** Retrieval-Augmented Generation · GenAI · Document Q&A · Enterprise AI

---

## Overview

Production RAG pipeline engineering for JPMorgan's LLM Suite — the bank's proprietary GenAI assistant deployed firm-wide in summer 2024, enabling 200,000 employees to query internal documents, generate content, and surface insights without exposing sensitive financial data to external AI providers. Recognized as **American Banker's 2025 Innovation of the Year**.

I held primary ownership of the RAG pipeline ML layer: document ingestion, chunking strategy design, embedding generation, vector store indexing, retrieval chain configuration, and the LLM output evaluation framework.

---

## Problem Statement

JPMorgan's knowledge workers spent an estimated 30–40% of daily work time on document retrieval, manual summarization, and information synthesis across disconnected internal repositories. The bank had banned external AI tools (ChatGPT, Gemini) over data privacy and compliance concerns, creating a competitive capability gap.

The technical challenge was not simply deploying an LLM — it was building a RAG architecture that could:
- Ingest a heterogeneous internal document corpus (regulatory filings, earnings decks, legal agreements, policy docs, client correspondence)
- Retrieve contextually relevant passages with high precision across diverse document types
- Generate responses with verifiable factual grounding meeting JPMorgan's model risk governance requirements
- Do all of this without ever sending sensitive financial data to external providers

---

## My Contributions

| Area | Role |
|------|------|
| LangChain RAG pipeline (RetrievalQA, Map-Reduce chains) | **Primary Owner** |
| Hybrid chunking strategy design and validation experiments | **Primary Owner** |
| Sentence Transformers embedding pipeline | **Primary Owner** |
| Pinecone production index configuration (namespace separation, metadata filtering) | **Primary Owner** |
| Dual-environment vector DB architecture (ChromaDB dev / Pinecone prod) | **Primary Owner** |
| 250-pair ground-truth evaluation dataset + hallucination measurement | **Primary Owner** |
| 6-template prompt engineering library | **Primary Owner** |
| FastAPI serving layer coordination | Contributor |

---

## Technical Architecture

```
Internal Documents
(regulatory filings · earnings decks · legal agreements · policy docs · client correspondence)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  INGESTION LAYER                                              │
│  LangChain document loaders (PDF + text)                     │
│  → NLP preprocessing (NLTK · spaCy NER · TF-IDF)            │
│  → Hybrid chunking strategy                                   │
│    ├── Paragraph-boundary (legal/regulatory) ─ +14 prec pts │
│    └── Fixed-size 1000t/200 overlap (general content)        │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  EMBEDDING + INDEXING LAYER                                   │
│  Sentence Transformers: all-MiniLM-L6-v2 (batch=64)         │
│  Metadata tagging: doc_type · date · source · access_tier    │
│  Pinecone (prod): cosine similarity · namespace-per-category │
│  ChromaDB (dev): identical interface, zero code changes      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                              │
│  LangChain RetrievalQA — targeted Q&A, single documents      │
│  LangChain Map-Reduce — synthesis over 80,000-token docs     │
│    Map: parallel GPT-3.5-turbo chunk summaries               │
│    Reduce: GPT-4 synthesizes final response (58% cost saving)│
│  Role-based access: namespace filtering prevents cross-dept   │
│  retrieval (zero cross-classification events in staging)      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  GENERATION + EVALUATION LAYER                                │
│  GPT-4 (primary) · GPT-3.5-turbo (map step)                 │
│  6 prompt templates (compliance · earnings · legal · policy   │
│    · client advisory · meeting summarization)                 │
│  RAGAS evaluation: 3.8% hallucination rate (target: <5%)     │
│  Mandatory source attribution on all responses                │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### Hybrid Chunking Strategy (Empirically Validated)
Before running experiments, the team assumed fixed-size chunking would be sufficient for all document types. Controlled precision@5 experiments across 120 documents and 90 queries disproved this:

- **Paragraph-boundary** outperformed fixed-size by **14 precision points** on legal and regulatory documents
- Root cause: JPMorgan's legal agreements contain multi-sentence obligation clauses. Fixed-size chunking split these mid-clause, producing retrieved chunks that stated an obligation without its qualifying conditions — exactly the failure mode compliance flagged as unacceptable
- **Fixed-size** outperformed by 6 points on general policy documents where analytical context spans multiple paragraphs
- **Decision:** Hybrid — paragraph-boundary for legal/regulatory, fixed-size for general content. This became the LLM Suite ingestion standard.

### Map-Reduce Chain for Long Documents
Regulatory filings frequently exceed 80,000 tokens — 5x GPT-4's context window. Standard RetrievalQA top-k retrieval surfaced semantically similar but contextually incomplete passages, producing misleading partial answers. Map-Reduce solution:
- **Map step:** parallel GPT-3.5-turbo summarizations over individual chunks (cheaper model for high-volume step)
- **Reduce step:** GPT-4 synthesizes the map outputs (expensive model reserved for high-stakes synthesis)
- **Result:** 58% API cost reduction vs. running GPT-4 over all chunks, coherent synthesis responses without truncation artifacts

### Dual-Environment Vector DB Architecture
A unified LangChain abstraction layer wraps both ChromaDB and Pinecone behind a single retriever API. Development and staging use ChromaDB (zero cost, local); production uses Pinecone. Zero application code changes required when promoting between environments — validated across 500 parity test queries.

### Compliance Prompt Template Refinement
Initial compliance Q&A templates produced responses citing the correct policy but omitting material qualifying conditions. GPT-4 was applying a brevity heuristic, treating conditions as supporting detail. Fix: explicit instruction — *"when citing a policy threshold, enumerate all conditions, exceptions, and effective dates mentioned in the source passage before providing any interpretation."* This produced complete, condition-inclusive responses in 94% of test cases.

---

## Results

### Retrieval Quality
| Metric | Result |
|--------|--------|
| Precision@5 improvement (legal/regulatory) | **+14 points** (paragraph-boundary vs. fixed-size) |
| Cross-classification retrieval events (staging) | **0** |
| Max document length handled (Map-Reduce) | **80,000 tokens** |

### LLM Output Quality
| Metric | Result |
|--------|--------|
| Hallucination rate (250-pair benchmark) | **3.8%** (target: <5% ✅) |
| GPT-4 exact-match accuracy (JPMorgan benchmark) | **0.89** |
| Prompt templates SME-validated | **6/6** (all >90% approval) |
| Compliance template condition-inclusive rate | **94%** |

### Pipeline Performance
| Metric | Result |
|--------|--------|
| End-to-end pipeline runtime (standard docs) | **<800ms** |
| ChromaDB → Pinecone promotion code changes | **0** |
| Staging parity test queries | 500 |

### Platform Impact
- **200,000 employees** onboarded within 8 months of launch
- **American Banker 2025 Innovation of the Year** — first GenAI platform deployed at scale across a major U.S. bank
- **$1.0–1.5B projected annual AI value** (per JPMorgan President Daniel Pinto) — LLM Suite one of three primary GenAI value-generation programs

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM Orchestration | LangChain (RetrievalQA, Map-Reduce, ConversationalRetrievalChain) |
| Vector Databases | ChromaDB (dev/staging), Pinecone (production, namespace-separated) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`), batch processing |
| LLM APIs | OpenAI GPT-4, GPT-3.5-turbo |
| Evaluation | Ground-truth Q&A dataset (250 pairs), RAGAS-style hallucination scoring, precision@5 |
| Serving & Data | FastAPI, PostgreSQL, SQLAlchemy, Pandas, Docker |
| Experiment Tracking | MLflow (pipeline config versioning, evaluation run logging) |

---

## Project Structure

```
2_LLM_Suite_RAG_Pipeline/
├── README.md
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py        # LangChain PDF + text loaders with metadata tagging
│   │   ├── chunking_strategy.py      # Hybrid chunking: paragraph-boundary + fixed-size
│   │   └── chunking_validator.py     # Controlled precision@5 experiments across strategies
│   ├── embeddings/
│   │   └── embedding_pipeline.py     # Sentence Transformers batch embedding pipeline
│   ├── vectorstore/
│   │   ├── pinecone_store.py         # Pinecone production config (namespace, metadata filter)
│   │   ├── chromadb_store.py         # ChromaDB dev config
│   │   └── store_factory.py          # Unified interface (zero code change env switching)
│   ├── retrieval/
│   │   ├── rag_chain.py              # RetrievalQA + Map-Reduce chain setup
│   │   └── map_reduce_chain.py       # Long-document synthesis chain
│   ├── evaluation/
│   │   ├── ground_truth_dataset.py   # 250-pair Q&A dataset construction
│   │   ├── hallucination_scorer.py   # Hallucination rate measurement
│   │   └── precision_evaluator.py    # Precision@5 evaluation framework
│   └── prompts/
│       ├── compliance_qa.py          # Compliance Q&A template (highest-stakes)
│       ├── earnings_analysis.py      # Earnings deck analysis template
│       ├── contract_review.py        # Contract review template
│       ├── policy_lookup.py          # Policy lookup template
│       ├── client_advisory.py        # Client advisory template
│       └── meeting_summary.py        # Meeting summarization template
├── notebooks/
│   ├── 01_chunking_validation.ipynb  # Precision@5 experiments across strategies
│   ├── 02_embedding_analysis.ipynb   # Embedding cluster visualization (UMAP)
│   ├── 03_retrieval_evaluation.ipynb # Retrieval quality benchmarks
│   ├── 04_hallucination_analysis.ipynb # 250-pair evaluation results
│   └── 05_prompt_iteration.ipynb     # Prompt engineering iteration log
├── evaluation/
│   └── ground_truth_qa.jsonl         # 250-pair annotated Q&A dataset (example format)
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Setup & Usage

```bash
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export PINECONE_API_KEY=your_key
export PINECONE_ENV=your_env

# Run in dev mode (ChromaDB, no cloud costs)
RAG_ENV=development python -m src.retrieval.rag_chain

# Run evaluation benchmark
python -m src.evaluation.hallucination_scorer \
    --dataset evaluation/ground_truth_qa.jsonl \
    --model gpt-4
```

### Quick Example

```python
from src.vectorstore.store_factory import get_vector_store
from src.retrieval.rag_chain import build_rag_chain

# Works identically in dev (ChromaDB) and prod (Pinecone)
vector_store = get_vector_store(env="development")
chain = build_rag_chain(vector_store, model="gpt-4")

response = chain.invoke({
    "query": "What is the current policy on derivatives reporting thresholds?"
})
print(response["answer"])
print(response["source_documents"])  # Mandatory source attribution
```

---

## Key Learnings

1. **Don't assume chunking strategy — run experiments.** Fixed-size chunking seemed sufficient before the data showed a 14-point precision gap on legal documents. The experiment cost a few days; the insight defined the platform's ingestion standard.

2. **Map-Reduce isn't just a cost optimization — it's an architectural requirement for long documents.** Top-k retrieval on 80,000-token documents reliably surfaces contextually similar but semantically incomplete passages. Map-Reduce is the correct solution.

3. **Hallucination measurement must be a hard gate, not advisory.** Every compliance prompt template was validated against the ground-truth dataset before deployment. The 9 hallucinated responses in the 250-pair benchmark were all peripheral contextual details — none involved material regulatory thresholds.

4. **Prompt engineering is iterative engineering, not art.** The compliance template took 4 iteration cycles and a compliance SME consultation to identify the root cause (GPT-4's brevity heuristic suppressing qualifying conditions). Systematic root cause analysis, not intuition, got to the fix.
