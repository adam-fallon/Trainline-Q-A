import shutil
import langchain

langchain.verbose = True
from tl_loaders.TrainlineTrainTimeLoader import TrainlineTrainTimeLoader
import jinja2
from time import sleep
from difflib import SequenceMatcher
from dotenv import load_dotenv

from main import create_llm, create_store, ask_question

load_dotenv()
# embedding_model = "sentence-transformers/all-mpnet-base-v2"
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
force_reindex = True
chat_history = []
eval_loop = 4

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

eval_questions = {
    "1": "what is the cheapest tickets to {{destination}}",
    "2": "when is the last train to {{destination}}",
    "3": "Journey time from {{origin}} to {{destination}}?",
    "4": "Name the trains and bus operators from {{origin}} to {{destination}}?",
    "5": "Frequency of trains per day from {{origin}} to {{destination}}",
    "6": "Distance from {{origin}} to {{destination}}",
    "7": "What is the price from {{origin}} to {{destination}}?",
    "8": "Are there changes from {{origin}} to {{destination}}?",
}

eval_answers_lookup = {
        "1": "Price from {{origin}} to {{destination}}",
        "2": "Last train from {{origin}} to {{destination}}",
        "3": "Journey time from {{origin}} to {{destination}}",
        "4": "Train operators from {{origin}} to {{destination}}",
        "5": "Frequency from {{origin}} to {{destination}}",
        "6": "Distance from {{origin}} to {{destination}}",
        "7": "Price from {{origin}} to {{destination}}",
        "8": "Changes from {{origin}} to {{destination}}"}


def get_od_pairs():
    routes = list(urls.values())
    od_pairs = []
    for r in routes:
        od_pairs.append({'origin': r.split(" to ")[0],
                         'destination': r.split(" to ")[1]})
    return od_pairs


def get_eval_questions(origin, destination):
    questions = []
    for k, v in eval_questions.items():
        questions.append({"od_pair": origin + " to " + destination,
                          "question_id": environment.from_string(k).render(origin=origin, destination=destination),
                          "question_text": environment.from_string(v).render(origin=origin, destination=destination),
                          "answer_text": get_eval_questions_and_answers(origin, destination).get(k)})
    return questions


def get_eval_questions_and_answers(origin, destination):
    fetched_answers = eval_answers.get(origin + " to " + destination)
    answers = {}
    for k, v in eval_answers_lookup.items():
        answers[k] = fetched_answers.get(environment.from_string(v).render(origin=origin, destination=destination), "errorNotFound")
    return answers


def flatten_and_summarise_eval_results(eval_score):
    flattened_results = {}
    for s in eval_score:
        question = flattened_results.get('question_id', s.get('question_id'))
        score = flattened_results.get('score', 0) + s.get('score')
        count = flattened_results.get('count', 0) + 1
        flattened_results.update(
            {question: {"score": score, "count": count, "percentage_score": round((score / count) * 100, 2)}})
    return flattened_results


def load_docs():
    loader = TrainlineTrainTimeLoader(list(urls.keys()), urls_to_od_pair=urls)
    html = loader.load()['referenceData']
    return html


def eval():
    questions_answers = []
    eval_score = []
    for od in get_od_pairs():
        questions_answers.extend(get_eval_questions(**od))
    for qa in questions_answers:
        for l in range(eval_loop):
            sleep(1)
            response = ask_question(qa.get("question_text"), llm=llm, db=db, answer_mode='eval')
            eval_score.append({"od_pair": qa.get("od_pair"),
                               "question_id": qa.get("question_id"),
                               "question_text": qa.get("question_text"),
                               "answer_text": qa.get("answer_text"),
                               "llm_answer": response,
                               "score": SequenceMatcher(None, qa.get("answer_text"), response).ratio()
                               })
    return eval_score, flatten_and_summarise_eval_results(eval_score)


if __name__ == "__main__":
    environment = jinja2.Environment()
    eval_answers = load_docs()
    print(eval_answers.get('London to Edinburgh'))
    if force_reindex:
        try:
            shutil.rmtree(persist_directory)
        except OSError as error:
            print(error)
    llm = create_llm()
    db = create_store()
    score, summary = eval()
    print(summary)
    print(score)
    print(eval_answers)
