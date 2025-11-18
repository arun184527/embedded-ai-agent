# 🧠 Embedded Systems AI Agent (Offline RAG + Local LLM)

A fully offline Embedded AI Assistant built using **Python**, **LlamaIndex**, **ChromaDB**, and **Ollama**.  
This agent can answer questions from your documentation, summarize files, retrieve technical info using RAG, and even automate embedded development tasks like building and flashing firmware.

---

## 🚀 Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Uses **SentenceTransformer embeddings**
- Stores vectors in **ChromaDB**
- Answers questions using your local documents
- Supports `raw` and `summary` query modes

### 🤖 Local LLM (Ollama)
- Runs completely offline
- Uses **tinydolphin** model for fast inference

### 🛠 Embedded Development Tools
- Build PlatformIO projects  
- Flash firmware  
- Read serial logs from microcontrollers  

### 📝 Debug Tools
- Includes a `debug_rag.py` tool to check what the RAG system retrieves  

---

## 📁 Project Structure