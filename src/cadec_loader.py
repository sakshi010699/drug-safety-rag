import re
import hashlib
from collections import Counter
from typing import Optional
from pathlib import Path
from schema import Document

DRUG_ALIASES = {
    # Atorvastatin
    "lipitor":            "atorvastatin",
    "atorvastatin":       "atorvastatin",

    # Diclofenac — all brand names and formulations
    "diclofenac":         "diclofenac",
    "voltaren":           "diclofenac",
    "voltaren-xr":        "diclofenac",
    "arthrotec":          "diclofenac",
    "cataflam":           "diclofenac",
    "cambia":             "diclofenac",
    "pennsaid":           "diclofenac",
    "solaraze":           "diclofenac",
    "zipsor":             "diclofenac",
    "flector":            "diclofenac",
    "diclofenac-sodium":  "diclofenac",
    "diclofenac-potassium": "diclofenac",
}
def normalize_drug_name(raw: str) -> str:
    return DRUG_ALIASES.get(raw.lower().strip(), raw.lower().strip())
    

def clean_text(text: str)-> str:
    # Unicode quotes and dashes → ASCII
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Remove any HTML tags that slipped through
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse 3+ newlines → 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def load_cadec(data_dir: Optional[str]= None)->list:
    if data_dir and Path(data_dir).exists():
        return _load_local(data_dir)
    print("No local CADEC found - using sample corpus.")
    print("For real CADEC: https://data.csiro.au/collection/csiro:10948")
    return _load_sample()

def _load_local(data_dir: str)->list:
    docs= []
    for path in sorted(Path(data_dir).glob("*.txt")):
        raw_drug = path.stem.split('.')[0]
        drug = normalize_drug_name(raw_drug)
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        if len(text)<20:
            continue
        docs.append(Document(
            text=text,
            drug_name=drug,
            source='cadec',
            metadata= {"filename": path.name}
        ))
    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs

def _load_sample()->list:
    docs = []
    for i, post in enumerate(SAMPLE_POSTS):
        docs.append(Document(
            text = post['text'],
            drug_name= post['drug'],
            source= 'cadec',
            metadata= {"post_index": i, "is_sample": True}
        ))
    print(f"Loaded {len(docs)} sample documents.")
    return docs