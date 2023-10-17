import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from tl_loaders.TrainlineTrainTimeLoader import TrainlineTrainTimeLoader
from langchain.chains import VectorDBQA
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


def ask_question(query):
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db)
    result = qa.run(query)
    return result


def setup_gradio():
    demo = gr.Interface(fn=ask_question, inputs="text", outputs="text")
    demo.launch()


def load_docs():
    urls = {
        "https://www.thetrainline.com/en/train-times/london-to-edinburgh": "london to edinburgh",
        "https://www.thetrainline.com/train-times/madrid-to-barcelona": "madrid to barcelona",
        "https://www.thetrainline.com/en/train-times/rome-to-madrid": "rome to madrid",
        "https://www.thetrainline.com/en/train-times/barcelona-to-madrid": "barcelona to madrid",
    }

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

    vectorstore = Chroma.from_documents(
        documents=create_splits(),
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=vector_store_name,
    )

    return vectorstore


def create_llm():
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_options)
    return llm


if __name__ == "__main__":
    llm = create_llm()
    db = create_store()
    setup_gradio()
