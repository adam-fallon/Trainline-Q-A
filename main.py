import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from tl_loaders.TrainlineTrainTimeLoader import TrainlineTrainTimeLoader
from langchain.chains import RetrievalQA
import gradio as gr

from dotenv import load_dotenv

load_dotenv()

embedding_model = "all-MiniLM-L6-v2"
persist_directory = "docs/chroma/"
chunk_size = 1000
chunk_overlap = 0

vector_store_name = "train_time_info"
vector_store_info = "Train time info"

repo_id = "google/flan-t5-xxl"

model_options = {"temperature": 0.1, "max_length": 256}
greet = "Ask a question in the like 'How many trains per day from Rome to Madrid'"

llm = None
db = None
force_reindex = False

# Few random ones and top results from https://www.thetrainline.com/train-times
urls = {
    "https://www.thetrainline.com/en/train-times/london-to-edinburgh": "london to edinburgh",
    "https://www.thetrainline.com/train-times/madrid-to-barcelona": "madrid to barcelona",
    "https://www.thetrainline.com/en/train-times/rome-to-madrid": "rome to madrid",
    "https://www.thetrainline.com/en/train-times/barcelona-to-madrid": "barcelona to madrid",
    "https://www.thetrainline.com/en/train-times/london-to-madrid": "london to madrid",
    "https://www.thetrainline.com/en/train-times/london-to-manchester": "london to manchester",
    "https://www.thetrainline.com/en/train-times/leeds-to-london": "leeds to london",
    "https://www.thetrainline.com/en/train-times/london-to-birmingham": "london to birmingham",
    "https://www.thetrainline.com/en/train-times/london-to-brighton": "london to brighton",
    "https://www.thetrainline.com/en/train-times/manchester-to-london": "manchester to london",
    "https://www.thetrainline.com/en/train-times/edinburgh-to-london": "edinburgh to london",
    "https://www.thetrainline.com/en/train-times/glasgow-to-manchester": "glasgow to manchester",
    "https://www.thetrainline.com/en/train-times/glasgow-to-liverpool": "glasgow to liverpool",
    "https://www.thetrainline.com/en/train-times/glasgow-to-leeds": "glasgow to leeds",
    "https://www.thetrainline.com/en/train-times/birmingham-to-glasgow": "birmingham to glasgow",
    "https://www.thetrainline.com/en/train-times/london-to-newcastle": "london to newcastle",
}


def ask_question(message, history):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
    )
    result = qa(message)
    source = result["source_documents"][0].metadata["source"]
    if result['result'] == "I don't know":
        return f"{result['result']}"
    else:
        return f"{result['result']}\n[Source]({source})"


def setup_gradio():
    demo = gr.ChatInterface(
        fn=ask_question,
        examples=[
            "Trains per day from London to Edinburgh?",
            "When is the last train from Madrid to Barcelona?",
            "Train and bus operators from Rome to Madrid?",
            "How many changes from Barcelona to Madrid?",
            "Price from London to Madrid?",
        ],
        title="Trainline Q & A 🤖",
        description=f"Ask questions about routes. Supported routes: {', '.join(urls.values())}",
    )

    demo.launch()


def load_docs():
    loader = TrainlineTrainTimeLoader(list(urls.keys()), urls_to_od_pair=urls)
    html = loader.load()

    return html


def create_splits():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(load_docs())
    return splits


def create_store():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    print("Attempting to load vector store from disk...")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    doc_count = len(vectorstore.get()["documents"])

    if doc_count <= 0 or force_reindex:
        print("Creating document store...")
        vectorstore = Chroma.from_documents(
            documents=create_splits(),
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print("Attempting to save vector store to disk...")
        vectorstore.persist()
        print("Done saving vector db to disk!")
    else:
        print(f"Loaded Vector DB from disk, Doc count: {doc_count}")

    return vectorstore


def create_llm():
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_options)
    return llm


if __name__ == "__main__":
    llm = create_llm()
    db = create_store()
    setup_gradio()
