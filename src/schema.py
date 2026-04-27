from dataclasses import dataclass, field
import hashlib
from typing import Optional


@dataclass
class Document:
    text: str
    source: str      # "cadec" | "fda_label" | "clinical_note"
    drug_name: str
    doc_id: str = ""
    metadata: dict =field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            content = f"{self.source}:{self.drug_name}:{self.text[:200]}"
            self.doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
    def word_count(self):
        return len(self.text.split())

    def __repr__(self):
        preview = self.text[:50].replace('\n', ' ')
        return (f"Document(id={self.doc_id}, source={self.source!r}, "
                f"drug={self.drug_name!r}, words={self.word_count()}, "
                f"preview={preview!r})")


@dataclass
class Chunk:
    text: str
    source: str
    doc_id: str
    drug_name: str
    chunk_index: int  # position within the document (0, 1, 2...)
    char_start: int # where the chuk starts in the original doc
    char_end: int   # where it ends
    chunk_id: str = ""
    vector_index: Optional[int]= None  # set later when added to FAISS
    metadata: dict= field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            content = f"{self.doc_id}:{self.chunk_index}:{self.char_start}"
            self.chunk_id = hashlib.md5(content.encode()).hexdigest()[:12]

    def is_indexed(self):
        return self.vector_index is not None

    def word_count(self):
        return len(self.text.split())

    def __repr__(self):
        preview = self.text[:50].replace('\n', ' ')
        status = f"vec={self.vector_index}" if self.is_indexed() else "not indexed"
        return (f"Chunk(id={self.chunk_id}, doc={self.doc_id}, "
                f"src={self.source!r}, drug={self.drug_name!r}, "
                f"idx={self.chunk_index}, {status}, "
                f"preview={preview!r})")



