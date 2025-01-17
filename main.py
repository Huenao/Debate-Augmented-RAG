import json
import os
import random
import time

import numpy as np
from config import *
from model import *


def init_output_file(args):
    save_path = os.path.join(args.output_dir, args.model, args.paradigm)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    output_file = os.path.join(save_path, f"{args.dataset}_{args.agents}Agents_{args.rag_agents}ragAgents_{args.rounds}Rounds_{args.question_cnts}q_{timestamp}.json")

    # create the output file if it does not exist
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("{}")
    else:
        raise ValueError("Output file already exists. Please delete the file or change the output file name.")
    return output_file


def append_dict(output_file, append_data):
    with open(output_file, "r+") as f:
        f.seek(0)
        data = json.load(f)

        data.update(append_data)

        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)


def main(args):
    start_time = time.time()  # Start timing

    output_file = init_output_file(args)
    
    append_dict(output_file, {"configs": vars(args)})

    # initialize LLM
    if "Llama" in args.model:
        llm_model = Llama(args.model, args.model_id, args.max_new_tokens, args.device)
    else:
        raise ValueError("Invalid model name.")
    
    # initialize debate process
    if args.paradigm == "DRAG":
        reason_process = DebateWithRAG(llm_model, args)
    elif args.paradigm == "single_agent":
        reason_process = SingleAgent(llm_model, args)
    elif args.paradigm == "cot":
        reason_process = CoT(llm_model, args)
    elif args.paradigm == "reflection":
        reason_process = Reflection(llm_model, args)
    else:
        raise ValueError("Invalid paradigm name.")
    
    # prepare questions
    selected_questions = prepare_questions(args)

    for q in selected_questions:
        question, answer = prepare_question_answer(args, q)

        print("****** Question ******")
        print(question)

        reason_history = reason_process.reason(question)
        # remember to clear the contexts for each agent after reasoning
        reason_process.clear_contexts()

        append_dict(output_file, {question: {"answer": answer, "agents_contexts": reason_history}})

    # Total time taken (hours)
    total_time = (time.time() - start_time) / 3600
    print("*"*10)
    print(f"Total time taken: {total_time} h")
    print("*"*10)
    append_dict(output_file, {"Runing Time": total_time})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    main(args)