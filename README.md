\# Drug Safety RAG — Built from Primitives



A Retrieval-Augmented Generation system for pharmacovigilance,

built without LangChain or LlamaIndex.



\## What it does

Answers drug safety questions by retrieving relevant evidence

from patient reviews, FDA drug labels, and clinical notes —

then generating a grounded answer citing its sources.



\## Why from scratch?

To understand RAG in depth.



\## Architecture

\[We'll fill this in as we build]



\## Data sources

\- CADEC corpus — patient forum posts with ADR annotations

\- openFDA API — official drug prescribing information  

\- Synthetic clinical notes — generated for evaluation



\## Stack

\- Embeddings: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased)

\- Vector store: FAISS (IndexFlatL2)

\- LLM: Claude API (direct HTTP, no SDK)

\- No LangChain. No LlamaIndex.



\## Previous work

Built on top of my ADR extraction project using BiLSTM and BioBERT.

https://github.com/sakshi010699/drug-adr-nlp

