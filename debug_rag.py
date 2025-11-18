# debug_rag.py
import chromadb
from local_embed import get_local_embedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.ollama import Ollama

def debug_query(query, persist_dir="db", top_k=5):
    embed_model = get_local_embedding()
    llm = Ollama(model="tinydolphin")

    # connect chroma
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    coll = chroma_client.get_or_create_collection("embedded_docs")

    print("\n🚀 COLLECTION INFO:", coll.count(), "items")

    # vector store + storage context
    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # run retrieval
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    print(f"\n🔍 Retrieved {len(nodes)} nodes")

    for i, n in enumerate(nodes):
        print(f"\n----- NODE {i+1} -----")
        try:
            print(n.get_text())
        except:
            print(n.text)
        print("----------------------")

if __name__ == "__main__":
    q = input("Query: ").strip()
    debug_query(q)
