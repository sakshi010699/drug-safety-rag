import re
from typing import Optional
from schema import Document, Chunk


ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof",
    "mg", "ml", "kg", "vs", "etc",
    "e.g", "i.e", "approx", "no",
}

def split_sentences(text: str) -> list:
    # Split on punctuation followed by space + capital letter
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = []
    buffer = ""
    for fragment in raw:
        if buffer:
            fragment = buffer + " "+ fragment
            buffer = ""
        words = fragment.rstrip().split()
        if words:
            last_word = words[-1].rstrip(".").lower()
            if last_word in ABBREVIATIONS:
                buffer= fragment
                continue
        if fragment.strip():
            sentences.append(fragment.strip())

    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences if sentences else [text.strip()]

def chunk_document(doc, chunk_size=400, overlap =1, min_chunk_size =80):
    sentences = split_sentences(doc.text)

    # Edge case: very short document fits in one chunk
    if not sentences or len(doc.text) <= chunk_size:
        return [Chunk(
            text= doc.text,
            drug_name = doc.drug_name,
            source = doc.source,
            doc_id = doc.doc_id,
            chunk_index = 0,
            char_start = 0,
            char_end = len(doc.text),
            
        )]
    chunks = []
    chunk_index = 0
    i=0 # sentence pointer

    while i < len(sentences):
        current = []
        current_len = 0
        j=i
        while j<len(sentences):
            s = sentences[j]
            # +1 for the space between sentences
            candidate_len = current_len + len(s) + (1 if current else 0)

            if candidate_len>chunk_size and current:
                break # this sentence would overflow — stop here

            current.append(s)
            current_len= candidate_len
            j=j+1            

        chunk_text = " ".join(current)
        # Skip tiny trailing fragments — merge upward instead

        if len(chunk_text)<min_chunk_size and chunks:
            prev= chunks[-1]
            merged = prev.text + " " + chunk_text
            chunks[-1] = Chunk(
                text=merged,
                doc_id=doc.doc_id,
                source=doc.source,
                drug_name=doc.drug_name,
                chunk_index=prev.chunk_index,
                char_start=prev.char_start,
                char_end=prev.char_start + len(merged),
            )
            break

        # Find character offset in original text
        char_start = doc.text.find(current[0]) if current else 0
        char_end = char_start + len(chunk_text) 

        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc.doc_id,
            source=doc.source,
            drug_name=doc.drug_name,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=min(char_end, len(doc.text)),
            metadata={"sentence_count": len(current)}
        ))

        chunk_index += 1

        advance = max(1, len(current) - overlap)
        i= i+advance
    return chunks