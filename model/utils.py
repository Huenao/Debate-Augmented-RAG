import re

def single_agent_pred_parse(dataset):
    FINAL_ANSWER_PREFIX = "The answer is:"
    for item in dataset:
        pred = item.pred
        if FINAL_ANSWER_PREFIX in pred:
            answer = pred.split(FINAL_ANSWER_PREFIX)[1].strip()
        else:
            answer = pred
        item.update_output('raw_pred', pred)
        item.update_output('pred', answer)
    return dataset