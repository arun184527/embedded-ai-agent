# config.py
from llama_index.llms.ollama import Ollama

def get_llm():
    return Ollama(model="tinydolphin")