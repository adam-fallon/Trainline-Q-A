# Trainline Train Time Q & A
Simple project using an LLM to do Q&A over Trainline Train Time Pages.

Retrieves some web content from Trainline, stores the Journey Info in a VectorDB and allows you to do question and answering over the data.

- Model: `meta-llama/Llama-2-13b-chat-hf`
```
model_options = {
    "temperature": 0.1,
    "max_length": 256,
    "stop_sequence": ".",
    "max_new_tokens": 2056,
}
```
- Embeddings: `text-embedding-ada-002-v2`
```
ChunkingStrategy = RecursiveCharacterTextSplitter
chunk_size = 1000
chunk_overlap = 0
```
- VectorDB: `ChromaDB`
```
search_kwargs = {
    "k": 5,
    "search_type": "mmr"
}
```

## Setup
1. `python -m venv env`
2. `source env/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py` (if you don't have the vector db populated already change `force_reindex` in `main.py` from `False` to `True`)
5. Go to http://localhost:7860 in your browser
