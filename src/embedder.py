from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

print("Loading tokenizer...")
tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading model...")
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval() # Disable dropout. We want determinsitic output at inference.

print("Model loaded.")
print("Modle hidden emdding size: ", model.config.hidden_size)
print("Max position embeddings: ", model.config.max_position_embeddings)

def encode(texts: list, batch_size: int)-> np.ndarray:
    all_embeddings= []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        # step 1 = tokenize
        encoded = tokenizer(
            batch,
            truncation= True,
            padding = True,
            max_length= 512,
            return_tensors = "pt"
        )

        # step 2 = Forward Pass
        with torch.no_grad():
            outputs = model(**encoded)

        # step 3 = mean pooling
        token_embeddings= outputs.last_hidden_state   # [batch_size, seq_len, 768]
        attention_mask = encoded["attention_mask"]    # [batch_size, seq_len]

        attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        sum_embeddings = (token_embeddings*attention_mask_expanded).sum(dim=1)  # [batch_size, 768]
        sum_mask = attention_mask_expanded.sum(dim =1).clamp(min = 1e-9)        # [batch_size, 1]

        pooled = sum_embeddings/sum_mask    # [batch_size, 768]


        # step 4 = L2 Normalize
        normed = F.normalize(pooled, p=2, dim=1)   # [batch, 768]


        all_embeddings.append(normed.numpy())


    return np.vstack(all_embeddings)