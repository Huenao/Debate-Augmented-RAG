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
    elif args.dataset == "HotpotQA":
        with open("./data/HotpotQA/hotpot_dev_fullwiki_v1.json", "r") as f:
            all_questions = json.load(f)
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
    elif args.dataset == "HotpotQA":
        return q["question"], q["answer"]
    
    raise ValueError("Invalid dataset name.")