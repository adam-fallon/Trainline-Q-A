import shutil
import langchain

langchain.verbose = True
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from tl_loaders.TrainlineTrainTimeLoader import TrainlineTrainTimeLoader
from langchain.chains import RetrievalQA
import gradio as gr

from dotenv import load_dotenv

load_dotenv()
embedding_model = "sentence-transformers/all-mpnet-base-v2"
persist_directory = "docs/chroma/openai"
chunk_size = 1000
chunk_overlap = 0

repo_id = "meta-llama/Llama-2-13b-chat-hf"

model_options = {
    "temperature": 0.1,
    "max_length": 256,
    "stop_sequence": ".",
    "max_new_tokens": 2056,
}
search_kwargs = {"k": 5, "search_type": "mmr"}

greet = "Ask a question in the like 'How many trains per day from Rome to Madrid'"

newline = "\n"
llm = None
db = None
force_reindex = False
chat_history = []

# Few random ones and top results from https://www.thetrainline.com/train-times
urls = {
    "https://www.thetrainline.com/en/train-times/london-to-edinburgh": "London to Edinburgh",
    "https://www.thetrainline.com/train-times/madrid-to-barcelona": "Madrid to Barcelona",
    "https://www.thetrainline.com/en/train-times/rome-to-madrid": "Rome to Madrid",
    "https://www.thetrainline.com/en/train-times/barcelona-to-madrid": "Barcelona to Madrid",
    "https://www.thetrainline.com/en/train-times/london-to-madrid": "London to Madrid",
    "https://www.thetrainline.com/en/train-times/london-to-manchester": "London to Manchester",
    "https://www.thetrainline.com/en/train-times/leeds-to-london": "Leeds to London",
    "https://www.thetrainline.com/en/train-times/london-to-birmingham": "London to Birmingham",
    "https://www.thetrainline.com/en/train-times/london-to-brighton": "London to Brighton",
    "https://www.thetrainline.com/en/train-times/manchester-to-london": "Manchester to London",
    "https://www.thetrainline.com/en/train-times/edinburgh-to-london": "Edinburgh to London",
    "https://www.thetrainline.com/en/train-times/glasgow-to-manchester": "Glasgow to Manchester",
    "https://www.thetrainline.com/en/train-times/glasgow-to-liverpool": "Glasgow to Liverpool",
    "https://www.thetrainline.com/en/train-times/glasgow-to-leeds": "Glasgow to Leeds",
    "https://www.thetrainline.com/en/train-times/birmingham-to-glasgow": "Birmingham to Glasgow",
    "https://www.thetrainline.com/en/train-times/london-to-newcastle": "London to Newcastle",
}


def ask_question(message, history):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
    )

    # result = qa(f"{newline.join(chat_history)}\n[INST]{message}[/INST]")
    result = qa(f"[INST]{message}[/INST]")

    answer = result["result"]
    print(result)

    chat_history.append(", ".join([message, answer]))

    if result["result"] == "I don't know":
        return answer
    else:
        try:
            source = result["source_documents"][0].metadata["source"]
            return f"{answer}\n[Source]({source})"
        except:
            return f"{answer}"


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
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings = OpenAIEmbeddings()

    print("Attempting to load vector store from disk...")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    doc_count = len(vectorstore.get()["documents"])
    print(doc_count)

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
    if force_reindex:
        try:
            shutil.rmtree(persist_directory)
        except OSError as error:
            print(error)
    llm = create_llm()
    db = create_store()
    setup_gradio()
