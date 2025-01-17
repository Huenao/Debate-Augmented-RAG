import json
import random
from glob import glob

import pandas as pd

MMLU_Supercategory = {
    "MMLU-STEM": ["algebra", "anatomy", "astronomy", "biology", "chemistry", "computer_science", "mathematics", "physics", "computer_security", "electrical_engineering", "statistics", "machine_learning"],
    "MMLU-Humanities": ["formal_logic", "history", "law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "world_religions"],
    "MMLU-Social Sciences": ["econometrics", "geography", "government_and_politics", "macroeconomics", "microeconomics", "psychology", "sexuality", "public_relations", "security_studies", "sociology", "us_foreign_policy"],
    "MMLU-Others": ["business", "clinical", "medicine", "global_facts", "human_aging", "management", "marketing", "medical_genetics", "miscellaneous", "nutrition", "accounting", "virology"]
}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def parse_df_question(df, ix):
    question_only = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "{}: (A) {}, (B) {}, (C) {}, (D) {} \n".format(question_only, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer


def parse_strategyqa_question(questions):
    question = questions["input"]
    answer = next(key for key, value in questions["target_scores"].items() if value == 1)

    return question, answer


def prepare_simple_ethical_question(questions):
    question_only = questions["input"]
    options = list(questions["target_scores"].keys())
    correct_index = next(i for i, value in enumerate(questions["target_scores"].values()) if value == 1)
    answer = (chr(65 + correct_index), options[correct_index])
    question = "{}: (A) {}, (B) {}, (C) {}, (D) {} \n".format(question_only, *options)
    return question, answer


def prepare_questions(args):
    if args.dataset == "MMLU":
        tasks = glob("./data/MMLU/test/*.csv")
        dfs = [pd.read_csv(task) for task in tasks]
        all_questions = [(df, idx) for df in dfs for idx in range(len(df))]
    elif args.dataset in ["MMLU-STEM", "MMLU-Humanities", "MMLU-Social Sciences", "MMLU-Others"]:
        dfs = []
        for name in MMLU_Supercategory[args.dataset]:
            tasks = glob(f"./data/MMLU/test/*{name}*.csv")
            dfs.extend([pd.read_csv(task) for task in tasks])
        all_questions = [(df, idx) for df in dfs for idx in range(len(df))]
    elif args.dataset == "MMLU-history":
        tasks = glob("./data/MMLU/test/*history*.csv")
        dfs = [pd.read_csv(task) for task in tasks]
        all_questions = [(df, idx) for df in dfs for idx in range(len(df))]
    elif args.dataset == "MMLU-psychology":
        tasks = glob("./data/MMLU/test/*psychology*.csv")
        dfs = [pd.read_csv(task) for task in tasks]
        all_questions = [(df, idx) for df in dfs for idx in range(len(df))]
    elif args.dataset == "MMLU-chemistry":
        tasks = glob("./data/MMLU/test/*chemistry*.csv")
        dfs = [pd.read_csv(task) for task in tasks]
        all_questions = [(df, idx) for df in dfs for idx in range(len(df))]
    elif args.dataset == "StrategyQA":
        with open("./data/StrategyQA/task.json", "r") as f:
            all_questions = json.load(f)["examples"]
    elif args.dataset == "SimpleEthical":
        with open("./data/Simple Ethical Questions/task.json", "r") as f:
            all_questions = json.load(f)["examples"]
    elif args.dataset == "GSM8K":
        print("Reading GSM8K test set...")
        all_questions = read_jsonl("./data/GSM8K/test.jsonl")
    elif args.dataset == "ASQA":
        print("Reading ASQA test set...")
        with open("./data/ASQA/ASQA.json", "r") as f:
            ASQA_dataset = json.load(f)
        dev_set = ASQA_dataset["dev"]
        train_set = ASQA_dataset["train"]
        all_questions = []
        for key in dev_set:
            new_dict = {
                "question": dev_set[key]["ambiguous_question"],
                "qa_pairs": dev_set[key]["qa_pairs"]
            }
            all_questions.append(new_dict)
    else:
        raise ValueError("Invalid dataset name.")
    
    print(f"Total number of {args.dataset} questions: ", len(all_questions))

    # check if there are enough unique questions to satisfy the request
    if args.question_cnts > len(all_questions):
        raise ValueError("Not enough unique questions to satisfy the request.")
    # randomly select questions
    selected_questions = random.sample(all_questions, k=args.question_cnts)

    return selected_questions


def prepare_question_answer(args, q):
    if "MMLU" in args.dataset:
        df = q[0]
        idx = q[1]
        return parse_df_question(df, idx)
    elif args.dataset == "StrategyQA":
        return parse_strategyqa_question(q)
    elif args.dataset == "SimpleEthical":
        return prepare_simple_ethical_question(q)
    elif args.dataset == "GSM8K":
        return q['question'], q['answer']
    elif args.dataset == "ASQA":
        return q["question"], q["qa_pairs"]
    
    raise ValueError("Invalid dataset name.")