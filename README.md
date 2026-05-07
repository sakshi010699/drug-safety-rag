# Drug Safety RAG — Built from Primitives

A Retrieval-Augmented Generation system for pharmacovigilance,
built without LangChain or LlamaIndex.

## Motivation
I built an ADR extraction project using BiLSTM 
and BioBERT, that led me to wonder how LLMs could answer questions 
about drug safety rather than just extract entities. 

## What it does
Answers drug safety questions by retrieving relevant evidence from 
patient reviews, then generating a grounded answer that cites its sources.

Example:

**Query:** "Does diclofenac affect blood pressure?"

**Answer:** Based on the patient reports, diclofenac may affect blood 
pressure. Report 1 mentions a slight effect on blood pressure with 
Cataflam. Report 4 specifically states the patient experienced increased 
BP while taking diclofenac...

The system refuses to answer when evidence is insufficient:

**Query:** "What are the side effects of Cymbalta?"

**Answer:** The available patient reports do not contain sufficient 
information to answer this question.

## Why from scratch
Most RAG tutorials use LangChain or LlamaIndex which abstract everything 
into three lines. Building from primitives means understanding every 
component and every failure mode:
- Chunking strategy directly affects retrieval quality
- Mean pooling with attention masking matters for embedding accuracy
- Normalisation consistency between indexing and query time is critical
- Prompt design determines whether the LLM grounds its answer or hallucinates

## Architecture

Raw Documents → Chunker → PubMedBERT Embedder → FAISS Index
                                                        ↓
User Query → Query Encoder → Similarity Search → Top-k Chunks
                                                        ↓
                              Prompt Builder → LLM API → Grounded Answer

## Components

**schema.py** — Document and Chunk dataclasses. Every component speaks 
this schema. Stable IDs derived from content hash for reproducibility.

**cadec_loader.py** — Loads CADEC corpus (1,220 patient forum posts). 
Handles brand name normalisation (Lipitor → atorvastatin, Arthrotec → 
diclofenac) and text cleaning.

**chunker.py** — Sentence-aware chunking with overlap. Never splits 
inside a sentence. Handles abbreviations (Dr., mg) that trip up generic 
splitters.

**embedder.py** — PubMedBERT encoder with mean pooling and L2 
normalisation. Trained from scratch on biomedical text — places 
"myalgia", "muscle pain", and "legs are killing me" in the same vector 
neighbourhood.

**retriever.py** — FAISS IndexFlatL2 with drug-aware filtering. Detects 
drug mentioned in query and restricts search to relevant chunks before 
running similarity search.

**prompt_builder.py** — Assembles context and query into a structured 
prompt with hard grounding constraint and citation requirement.

**llm_client.py** — Direct HTTP call to LLM API. No SDK wrapper.

## Data
- **CADEC** (CSIRO Adverse Drug Event Corpus) — 1,220 annotated patient 
  forum posts, 2,549 chunks after processing
- Covers: atorvastatin (Lipitor) and diclofenac (Voltaren, Arthrotec, 
  Cataflam and other brand names)

## Stack
- Embeddings: PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased)
- Vector store: FAISS IndexFlatL2
- LLM: Llama 3 via Groq API (direct HTTP, no SDK)
- No LangChain. No LlamaIndex. No abstractions.

## Results
- 2,549 chunks indexed across 1,220 documents
- Drug-filtered retrieval prevents cross-drug contamination
- Grounding constraint prevents hallucination on out-of-corpus queries
- Answers cite specific patient reports by number

## Previous work
This project builds on my ADR extraction work using BiLSTM and BioBERT 
for named entity recognition on clinical text.
[https://github.com/sakshi010699/drug-adr-nlp]

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/drug-safety-rag
cd drug-safety-rag
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
jupyter notebook notebooks/05_prompt_and_llm.ipynb
```

## Author
Sakshi Jain