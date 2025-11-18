# embedded_agent.py
import os
import serial
import subprocess
import chromadb

from config import get_llm
from local_embed import get_local_embedding

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore


# ---------- Load RAG Index ----------
def load_index(persist_dir="db"):
    embed_model = get_local_embedding()

    from llama_index.llms.ollama import Ollama
    llm = Ollama(model="tinydolphin")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection("embedded_docs")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    from llama_index.core.storage.storage_context import StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index, llm


# ---------- Tools ----------
def build_project(path):
    try:
        result = subprocess.run(
            ["pio", "run"],
            cwd=path,
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def flash_project(path):
    try:
        result = subprocess.run(
            ["pio", "run", "-t", "upload"],
            cwd=path,
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def read_serial(port, baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=3)
        data = ser.read(2000).decode(errors="ignore")
        ser.close()
        return data
    except Exception as e:
        return str(e)


# ---------- Agent Loop ----------
def main():

    try:
        index, llm = load_index()
        retriever = index.as_retriever(similarity_top_k=3)
        rag_ready = True
        print("✔ RAG index loaded successfully.")
    except Exception as e:
        import traceback
        print("⚠ Failed to load RAG index:")
        traceback.print_exc()
        rag_ready = False

    print("\n✔ Embedded Systems Agent Ready (tinydolphin)\n")
    print("Commands:\n"
          "  ask <question>\n"
          "  ask raw <question>\n"
          "  ask summary <question>\n"
          "  build <project path>\n"
          "  flash <project path>\n"
          "  serial <port>\n"
          "  exit\n")

    while True:
        user = input(">>> ").strip()

        if user == "exit":
            break

        if user.startswith("build "):
            print(build_project(user.split(" ", 1)[1]))
            continue

        if user.startswith("flash "):
            print(flash_project(user.split(" ", 1)[1]))
            continue

        if user.startswith("serial "):
            print(read_serial(user.split(" ", 1)[1]))
            continue

        # ---------- ASK RAW MODE ----------
        if user.startswith("ask raw "):
            q = user[len("ask raw "):]

            if rag_ready:
                nodes = retriever.retrieve(q)
                print("\n📘 RAW Answer:\n")
                for n in nodes:
                    try:
                        print(n.get_text())
                    except:
                        print(n.text)
                print("\n")
            continue

        # ---------- ASK SUMMARY MODE ----------
        if user.startswith("ask summary "):
            q = user[len("ask summary "):]

            if rag_ready:
                nodes = retriever.retrieve(q)

                combined = "\n".join([n.get_text() if hasattr(n, "get_text") else n.text for n in nodes])

                summary_prompt = f"Summarize this in simple words:\n\n{combined}"
                result = llm.complete(summary_prompt)

                print("\n📘 Summary:\n", result, "\n")
            continue

        # ---------- DEFAULT ASK MODE ----------
        if user.startswith("ask "):
            q = user[len("ask "):]

            if rag_ready:
                nodes = retriever.retrieve(q)
                print("\n📘 Answer:\n")
                for n in nodes:
                    try:
                        print(n.get_text())
                    except:
                        print(n.text)
                print("\n")
            continue

        print("Unknown command. Use: ask, ask raw, ask summary, build, flash, serial")


if __name__ == "__main__":
    main()
