import argparse
import json
import os

from html4vision import Col, imagetable


def main(args):
    with open(os.path.join(args.file_path, "intermediate_data.json"), 'r') as f:
        data = json.load(f)
        
    id_col = []
    question_col = []
    gold_answer_col = []
    prompt_col = []
    raw_pred_col = []
    pred_col = []
    metric_score_col = []
    for d in data:
        id_col.append(d['id'])
        question_col.append(d['question'])
        gold_answer_col.append(json.dumps(d['golden_answers'], indent=4, ensure_ascii=False, sort_keys=True))
        prompt_col.append(d['output']['prompt'])
        raw_pred_col.append(d['output']['raw_pred'])
        pred_col.append(d['output']['raw_pred'])
        metric_score_col.append(json.dumps(d['output']['metric_score'], indent=4, ensure_ascii=False, sort_keys=True))
    
    # Retrieval results
    num_retrievals = len(data[0]["output"]["retrieval_result"])
    retrieval_result_lists = [[] for _ in range(num_retrievals)]
    for item in data:
        for idx, retrieval in enumerate(item["output"]["retrieval_result"]):
            retrieval_result_lists[idx].append(retrieval["contents"])
    
    columns = [
        Col("text", "ID", id_col), 
        Col("text", "Question", question_col),
        Col("text", "Answer", gold_answer_col),
        Col("text", "Metric Score", metric_score_col),
        Col("text", "Prediction", pred_col),
        Col("text", "Prompt", prompt_col),
        Col("text", "Raw Prediction", raw_pred_col),
    ]
    
    for idx, retrieval_result_list in enumerate(retrieval_result_lists):
        columns.append(Col("text", f"Retrieval Result {idx}", retrieval_result_list))
    
    folder_name = os.path.basename(os.path.normpath(args.file_path))
    
    imagetable(columns,
        out_file=os.path.join(args.file_path, f'{folder_name}.html'),
        sortable=True,
        sticky_header=True,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the json file containing the responses')
    args = parser.parse_args()

    main(args)