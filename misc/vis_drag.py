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
        id_col.extend([d['id']]*3)
        question_col.extend([d['question']]*3)
        gold_answer_col.extend([json.dumps(d['golden_answers'], indent=4, ensure_ascii=False, sort_keys=True)]*3)
        prompt_col.extend([d['output']['answer_input_prompt']]*3)
        raw_pred_col.extend([d['output']['raw_pred']]*3)
        pred_col.extend([d['output']['pred']]*3)
        metric_score_col.extend([json.dumps(d['output']['metric_score'], indent=4, ensure_ascii=False, sort_keys=True)]*3)
    
    agent_names = []
    QueryStage_Round0_InputPrompt = []
    QueryStage_Round0_Output = []
    QueryStage_Round1_InputPrompt = []
    QueryStage_Round1_Output = []
    QueryStage_Round2_InputPrompt = []
    QueryStage_Round2_Output = []
    AnswerStage_Round0_InputPrompt = []
    AnswerStage_Round0_Output = []
    AnswerStage_Round1_InputPrompt = []
    AnswerStage_Round1_Output = []
    AnswerStage_Round2_InputPrompt = []
    AnswerStage_Round2_Output = []
    
    for d in data:
        agent_names.append("Proponent Agent 0")
        QueryStage_Round0_InputPrompt.append(d['output'].get('QueryStage_Proponent Agent 0_Round0_InputPrompt', ""))
        QueryStage_Round0_Output.append(d['output'].get('QueryStage_Proponent Agent 0_Round0_Output', ""))
        QueryStage_Round1_InputPrompt.append(d['output'].get('QueryStage_Proponent Agent 0_Round1_InputPrompt', ""))
        QueryStage_Round1_Output.append(d['output'].get('QueryStage_Proponent Agent 0_Round1_Output', ""))
        QueryStage_Round2_InputPrompt.append(d['output'].get('QueryStage_Proponent Agent 0_Round2_InputPrompt', ""))
        QueryStage_Round2_Output.append(d['output'].get('QueryStage_Proponent Agent 0_Round2_Output', ""))
        AnswerStage_Round0_InputPrompt.append(d['output'].get('AnswerStage_Proponent Agent 0_Round0_InputPrompt', ""))
        AnswerStage_Round0_Output.append(d['output'].get('AnswerStage_Proponent Agent 0_Round0_Output', ""))
        AnswerStage_Round1_InputPrompt.append(d['output'].get('AnswerStage_Proponent Agent 0_Round1_InputPrompt', ""))
        AnswerStage_Round1_Output.append(d['output'].get('AnswerStage_Proponent Agent 0_Round1_Output', ""))
        AnswerStage_Round2_InputPrompt.append(d['output'].get('AnswerStage_Proponent Agent 0_Round2_InputPrompt', ""))
        AnswerStage_Round2_Output.append(d['output'].get('AnswerStage_Proponent Agent 0_Round2_Output', ""))

        agent_names.append("Opponent Agent 0")
        QueryStage_Round0_InputPrompt.append(d['output'].get('QueryStage_Opponent Agent 0_Round0_InputPrompt', ""))
        QueryStage_Round0_Output.append(d['output'].get('QueryStage_Opponent Agent 0_Round0_Output', ""))
        QueryStage_Round1_InputPrompt.append(d['output'].get('QueryStage_Opponent Agent 0_Round1_InputPrompt', ""))
        QueryStage_Round1_Output.append(d['output'].get('QueryStage_Opponent Agent 0_Round1_Output', ""))
        QueryStage_Round2_InputPrompt.append(d['output'].get('QueryStage_Opponent Agent 0_Round2_InputPrompt', ""))
        QueryStage_Round2_Output.append(d['output'].get('QueryStage_Opponent Agent 0_Round2_Output', ""))
        AnswerStage_Round0_InputPrompt.append(d['output'].get('AnswerStage_Opponent Agent 0_Round0_InputPrompt', ""))
        AnswerStage_Round0_Output.append(d['output'].get('AnswerStage_Opponent Agent 0_Round0_Output', ""))
        AnswerStage_Round1_InputPrompt.append(d['output'].get('AnswerStage_Opponent Agent 0_Round1_InputPrompt', ""))
        AnswerStage_Round1_Output.append(d['output'].get('AnswerStage_Opponent Agent 0_Round1_Output', ""))
        AnswerStage_Round2_InputPrompt.append(d['output'].get('AnswerStage_Opponent Agent 0_Round2_InputPrompt', ""))
        AnswerStage_Round2_Output.append(d['output'].get('AnswerStage_Opponent Agent 0_Round2_Output', ""))

        agent_names.append("Moderator")
        QueryStage_Round0_InputPrompt.append(d['output'].get('QueryStage_Moderator_Round0_InputPrompt', ""))
        QueryStage_Round0_Output.append(d['output'].get('QueryStage_Moderator_Round0_Output', ""))
        QueryStage_Round1_InputPrompt.append(d['output'].get('QueryStage_Moderator_Round1_InputPrompt', ""))
        QueryStage_Round1_Output.append(d['output'].get('QueryStage_Moderator_Round1_Output', ""))
        QueryStage_Round2_InputPrompt.append(d['output'].get('QueryStage_Moderator_Round2_InputPrompt', ""))
        QueryStage_Round2_Output.append(d['output'].get('QueryStage_Moderator_Round2_Output', ""))
        AnswerStage_Round0_InputPrompt.append(d['output'].get('AnswerStage_Moderator_Round0_InputPrompt', ""))
        AnswerStage_Round0_Output.append(d['output'].get('AnswerStage_Moderator_Round0_Output', ""))
        AnswerStage_Round1_InputPrompt.append(d['output'].get('AnswerStage_Moderator_Round1_InputPrompt', ""))
        AnswerStage_Round1_Output.append(d['output'].get('AnswerStage_Moderator_Round1_Output', ""))
        AnswerStage_Round2_InputPrompt.append(d['output'].get('AnswerStage_Moderator_Round2_InputPrompt', ""))
        AnswerStage_Round2_Output.append(d['output'].get('AnswerStage_Moderator_Round2_Output', ""))

    
    columns = [
        Col("text", "ID", id_col), 
        Col("text", "Question", question_col),
        Col("text", "Answer", gold_answer_col),
        Col("text", "Metric Score", metric_score_col),
        Col("text", "Agent", agent_names),
        Col("text", "Prediction", pred_col),
        Col("text", "Answer Stage Prompt", prompt_col),
        Col("text", "Raw Prediction", raw_pred_col),
        Col("text", "Query_Round0_InputPrompt", QueryStage_Round0_InputPrompt),
        Col("text", "Query_Round0_Output", QueryStage_Round0_Output),
        Col("text", "Query_Round1_InputPrompt", QueryStage_Round1_InputPrompt),
        Col("text", "Query_Round1_Output", QueryStage_Round1_Output),
        Col("text", "Query_Round2_InputPrompt", QueryStage_Round2_InputPrompt),
        Col("text", "Query_Round2_Output", QueryStage_Round2_Output),
        Col("text", "Answer_Round0_InputPrompt", AnswerStage_Round0_InputPrompt),
        Col("text", "Answer_Round0_Output", AnswerStage_Round0_Output),
        Col("text", "Answer_Round1_InputPrompt", AnswerStage_Round1_InputPrompt),
        Col("text", "Answer_Round1_Output", AnswerStage_Round1_Output),
        Col("text", "Answer_Round2_InputPrompt", AnswerStage_Round2_InputPrompt),
        Col("text", "Answer_Round2_Output", AnswerStage_Round2_Output),
    ]
    
    
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