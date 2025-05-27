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
    raw_pred_col = []
    pred_col = []
    metric_score_col = []
    for d in data:
        id_col.extend([d['id']]*3)
        question_col.extend([d['question']]*3)
        gold_answer_col.extend([json.dumps(d['golden_answers'], indent=4, ensure_ascii=False, sort_keys=True)]*3)
        raw_pred_col.extend([d['output']['raw_pred']]*3)
        pred_col.extend([d['output']['pred']]*3)
        metric_score_col.extend([json.dumps(d['output']['metric_score'], indent=4, ensure_ascii=False, sort_keys=True)]*3)
    
    agent_names = []
    Round0_InputPrompt = []
    Round0_Output = []
    Round1_InputPrompt = []
    Round1_Output = []
    Round2_InputPrompt = []
    Round2_Output = []
    
    for d in data:
        agent_names.append("Agent 1")
        Round0_InputPrompt.append(d['output'].get('Agent 1_Round_0_input_prompt', ""))
        Round0_Output.append(d['output'].get('Agent 1_Round_0_output', ""))
        Round1_InputPrompt.append(d['output'].get('Agent 1_Round_1_input_prompt', ""))
        Round1_Output.append(d['output'].get('Agent 1_Round_1_output', ""))
        Round2_InputPrompt.append(d['output'].get('Agent 1_Round_2_input_prompt', ""))
        Round2_Output.append(d['output'].get('Agent 1_Round_2_output', ""))
        
        agent_names.append("Agent 2")
        Round0_InputPrompt.append(d['output'].get('Agent 2_Round_0_input_prompt', ""))
        Round0_Output.append(d['output'].get('Agent 2_Round_0_output', ""))
        Round1_InputPrompt.append(d['output'].get('Agent 2_Round_1_input_prompt', ""))
        Round1_Output.append(d['output'].get('Agent 2_Round_1_output', ""))
        Round2_InputPrompt.append(d['output'].get('Agent 2_Round_2_input_prompt', ""))
        Round2_Output.append(d['output'].get('Agent 2_Round_2_output', ""))
        
        agent_names.append("Moderator")
        Round0_InputPrompt.append(d['output'].get('Moderator_Round_0_input_prompt', ""))
        Round0_Output.append(d['output'].get('Moderator_Round_0_output', ""))
        Round1_InputPrompt.append(d['output'].get('Moderator_Round_1_input_prompt', ""))
        Round1_Output.append(d['output'].get('Moderator_Round_1_output', ""))
        Round2_InputPrompt.append(d['output'].get('Moderator_Round_2_input_prompt', ""))
        Round2_Output.append(d['output'].get('Moderator_Round_2_output', ""))

    
    columns = [
        Col("text", "ID", id_col), 
        Col("text", "Question", question_col),
        Col("text", "Answer", gold_answer_col),
        Col("text", "Metric Score", metric_score_col),
        Col("text", "Agent", agent_names),
        Col("text", "Prediction", pred_col),
        Col("text", "Raw Prediction", raw_pred_col),
        Col("text", "Round 0 Input Prompt", Round0_InputPrompt),
        Col("text", "Round 0 Output", Round0_Output),
        Col("text", "Round 1 Input Prompt", Round1_InputPrompt),
        Col("text", "Round 1 Output", Round1_Output),
        Col("text", "Round 2 Input Prompt", Round2_InputPrompt),
        Col("text", "Round 2 Output", Round2_Output),
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