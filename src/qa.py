import json

import sys
sys.path.append('../src')
from agent import Agent, top_norm_prob
import numpy as np
import multiprocess as mp
from tqdm import tqdm


def QA(question:str, choices:list):
    qa_agent = Agent()
    str_choices = zip(range(len(choices)), choices)
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    str_choices = "\n".join([f"{alpha[idx]}. {choice}" for idx, choice in str_choices])
    qa_agent.load_message([{"role": "user", "content": f"# Question: \n{question} # Choices: \n{str_choices}"}])
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "multiple_choice_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "response": {
                "type": "string",
                "description": "The letter corresponding to the answer.",
                "enum": [alpha[idx] for idx in range(len(choices))]
            }
            },
            "required": [
                "response"
            ],
            "additionalProperties": False
        }
        }
    }
    response_json, response_logprobs = qa_agent.get_response(response_format = response_format, logprobs = True, temperature = 0)
    response = json.loads(response_json)["response"]
    top_prob = top_norm_prob(response_logprobs, response)
    top_prob_list = [(x.token, round(np.exp(x.logprob), 4)) for x in list(response_logprobs[3].top_logprobs)][:len(choices)]
    return response, top_prob, top_prob_list


def process_sample(ele):
    """Process a single MMLU sample."""
    question = ele['question']
    choices = ele['options']
    pred, prob, topk = QA(question, choices)  # Assuming QA is defined elsewhere
    ele['r_org'] = pred
    ele['p_r_org'] = prob
    ele['topk_org'] = topk
    return ele  # Return updated sample

def mmlu_eval(mmlu_samples, num_workers=mp.cpu_count()):
    """Evaluate MMLU samples using multiprocess for parallel execution with tqdm."""
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, mmlu_samples), total=len(mmlu_samples), desc="Processing MMLU"))
    return results  # Return processed samples