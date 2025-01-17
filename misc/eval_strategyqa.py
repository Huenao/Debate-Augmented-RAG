import argparse
import json
import re
from collections import Counter

import numpy as np


def parse_answer(input_str):
    """
    Extract the answer from the input string using regex.
    Args:
        input_str (str): Input string containing the answer.
    Returns:
        str: Parsed answer as an uppercase character, or None if not found.
    """
    pattern = r'The answer is \([A-B]\) (Yes|No)\.'
    matches = re.search(pattern, input_str)

    if matches:
        return matches.group(1)
    else:
        print(f"No valid answer found in: {input_str}")  # Print unmatched content for debugging
        return None

def compute_accuracy(gt, pred_solutions):
    """
    Compute accuracy by comparing ground truth to predicted answers.
    Args:
        gt (str): Ground truth answer.
        pred_solutions (list or str): Predicted solutions, either as a list or a single string.
    Returns:
        float: Accuracy (1 or 0).
    """
    if isinstance(pred_solutions, list):
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if not pred_answers:  # Check if pred_answers is empty
            return 0

        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer(pred_solutions)

    return 1 if gt == pred_answer else 0


def most_frequent(answers):
    """
    Find the most frequent answer in a list.
    Args:
        answers (list): List of answers.
    Returns:
        str: Most frequent answer, or None if the list is empty.
    """
    return Counter(answers).most_common(1)[0][0] if answers else None


def parse_args():
    """
    Parse command-line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the json file containing the responses", required=True)
    parser.add_argument("--use_summary", type=bool, help="Whether to use the summary agent responses",
                        default=True)
    parser.add_argument("--eval_round", type=int, help="Evaluation round", 
                        default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with open(args.file_path, "r") as file:
            response_dict = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading the JSON file: {e}")
    
    special_keys = ["configs", "Runing Time"]
    
    questions = [key for key in response_dict.keys() if key not in special_keys]
    
    print(f"Number of questions: {len(questions)}")

    accuracies = []

    for question in questions:
        question_data = response_dict[question]
        agents_response = question_data.get("agents_contexts")
        answer = question_data.get("answer")

        if not agents_response or not answer:
            raise ValueError(f"Missing data for question: {question}")
        
        agents = list(agents_response.keys())

        if args.use_summary:
            agent = agents[-1]  # last agent is the summary agent
            pred_solution = agents_response[agent][(args.eval_round * 2 - 1)]['content']
            accurate = compute_accuracy(answer, pred_solution)
        else:
            pred_solutions = []
            for agent in agents[:-1]:  # exclude the summary agent
                if len(agents_response[agent]) < args.eval_round * 2:
                    raise ValueError(f"Not enough rounds for evaluation in question: {question}")
                pred_solution = agents_response[agent][(args.eval_round * 2 - 1)]['content']
                pred_solutions.append(pred_solution)
            accurate = compute_accuracy(answer, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))

     # Print mean and standard error of accuracies
    mean_accuracy = np.mean(accuracies)
    std_error = np.std(accuracies) / np.sqrt(len(accuracies))
    print(f"Accuracy: {mean_accuracy:.4f} \u00b1 {std_error:.4f}")

if __name__ == "__main__":
    main()