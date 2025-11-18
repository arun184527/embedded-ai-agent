# local_embed.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_local_embedding():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")