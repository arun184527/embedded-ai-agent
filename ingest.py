# ingest.py
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from local_embed import get_local_embedding


def ingest_documents():
    print("Reading documents...")
    docs = SimpleDirectoryReader("docs").load_data()
    print(f"Loaded {len(docs)} documents. Creating embeddings...")

    embed_model = get_local_embedding()

    # Create / load chroma DB
    chroma_client = chromadb.PersistentClient(path="db")
    collection = chroma_client.get_or_create_collection("embedded_docs")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Store documents + embeddings
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("Ingestion completed. DB saved to: db")


if __name__ == "__main__":
    ingest_documents()
