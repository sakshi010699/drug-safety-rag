# embedder.py
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# These are None until load_model() is called
tokenizer = None
model = None

# Force CPU — avoids MPS/numpy conversion crash on Apple Silicon
DEVICE = torch.device("cpu")

def load_model():
    global tokenizer, model
    if model is not None:
        print("Model already loaded.")
        return
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading model...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}.")

def encode(texts: list, batch_size: int = 32) -> np.ndarray:
    if model is None:
        raise RuntimeError("Call load_model() before encode()")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        # Move inputs to same device as model
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        normed = F.normalize(pooled, p=2, dim=1)

        # .cpu() ensures tensor is on CPU before .numpy()
        all_embeddings.append(normed.cpu().numpy())

    return np.vstack(all_embeddings)