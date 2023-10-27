import shutil
import langchain

langchain.verbose = True
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from tl_loaders.TrainlineTrainTimeLoader import TrainlineTrainTimeLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

from dotenv import load_dotenv

load_dotenv()
embedding_model = "sentence-transformers/all-mpnet-base-v2"
persist_directory = "docs/chroma/"
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
    "https://www.thetrainline.com/en/train-times/glasgow-to-manchester": "Glasgow to Manchester",
    "https://www.thetrainline.com/en/train-times/glasgow-to-liverpool": "Glasgow to Liverpool",
    "https://www.thetrainline.com/en/train-times/glasgow-to-leeds": "Glasgow to Leeds",
    "https://www.thetrainline.com/en/train-times/birmingham-to-glasgow": "Birmingham to Glasgow",
    "https://www.thetrainline.com/en/train-times/london-to-newcastle": "London to Newcastle",
    "https://www.thetrainline.com/train-times/seville-to-madrid": "Seville to Madrid",
}


def ask_question(message):
    prompt = """
Use the following pieces of context to answer the question at the end. 
Be very succint and give just the answer - no other info.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
DO NOT RAMBLE or try to infer information that isn't in the context.
Take a deep breath and work on this problem step-by-step.
The prompt with have a to station and a from station - only answer if the exact station matches the context - say I don't know if the info isn't in the context.
The order is important. E.g Seville to Madrid is not the same as Madrid to Seville so if the order of stations in the context doesn't match the query then don't try answer.

{context}


Question: {question}
Helpful answer:
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt,
                input_variables=["context", "question"],
            ),
        },
    )

    print(qa.combine_documents_chain.llm_chain.prompt.template)

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
            doc = result["source_documents"][0].page_content
            return f"{answer}\n---\nMost relevant document: {doc}\nSource: {source}"
        except:
            return f"{answer}"


def setup_gradio():
    desc = """
    Welcome to Train Route Q&A! 

    A silly little demo of how RAG can ground the answers from an LLM.
    
    - This isn't an answer everything box - Really only expect grounded answers from the list of supported questions below.    
    - NOT A chat model - it is 1 response to 1 question so you can't ask follow up.
    - Data accurate as of 19th October 2023.
    - Always returns a source document most relevant to your question - even if the answer is not found.
    - 'Weak' models (like llama2-13b) won't listen to prompt rules effectively - clone the space and use a mode powerful model for best experience.
    
    Not an official Trainline Project!
    """

    long_desc = """
    Ask a question in like 'How many trains per day from Rome to Madrid'.     

    This info is scrapped from the table on train time pages. Example page here: https://www.thetrainline.com/train-times/manchester-to-london
    
    Not all train time pages have been scraped so you don't get answers for every route, just the ones in the supported routes below and only answers for the supported questions.
    
    Supported questions
    ---
    - Price from X to Y
    - Last train from X to Y
    - First train from X to Y
    - Frequency of trains from X to Y
    - Price of trains from X to Y
    - Operators of trains and buses from X to Y
    - Distance from X to Y
    - Number of Changes from X to Y
    - Journey time from X to Y

    Supported routes
    ---
    - London to Edinburgh
    - Madrid to Barcelona
    - Rome to Madrid
    - Barcelona to Madrid
    - London to Madrid
    - London to Manchester
    - Leeds to London
    - London to Birmingham
    - London to Brighton
    - Glasgow to Manchester
    - Glasgow to Liverpool
    - Glasgow to Leeds
    - Birmingham to Glasgow
    - London to Newcastle
    - Seville to Madrid

    Info
    ---
    - Model: meta-llama/Llama-2-13b-chat-hf
    - Embedding Model: sentence-transformers/all-mpnet-base-v2

    Prompt
    ---
    Use the following pieces of context to answer the question at the end. 
    Be very succint and give just the answer - no other info.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    DO NOT RAMBLE or try to infer information that isn't in the context.
    Take a deep breath and work on this problem step-by-step.
    The prompt with have a to station and a from station - only answer if the exact station matches the context - say I don't know if the info isn't in the context.
    The order is important. E.g Seville to Madrid is not the same as Madrid to Seville so if the order of stations in the context doesn't match the query then don't try answer.
    
    {context}
    
    Question: {question}
    Helpful answer:
    """

    iface = gr.Interface(
        ask_question,
        inputs="text",
        outputs="text",
        allow_screenshot=False,
        allow_flagging=False,
        description=desc, 
        article=long_desc
    )
    iface.launch()


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


if force_reindex:
    try:
        shutil.rmtree(persist_directory)
    except OSError as error:
        print(error)
llm = create_llm()
db = create_store()
setup_gradio()