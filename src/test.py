import sys
sys.path.append('src')
import pickle
from agent import Agent, top_norm_prob, create_openai_client
import qa
llm = "qwen-2.5-72B"
client = create_openai_client(llm)
agent = Agent(client, llm)
with open(f'data/MMLU-Pro/sample_results/org.pkl', 'rb') as f:
    res_org = pickle.load(f)


# results = qa.qa_eval_org(res_org[:100], llm=llm)
# print(results)

ele = res_org[10]

question = ele['question']
choices = ele['options']
str_choices = zip(range(len(choices)), choices)
alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
str_choices = "\n".join([f"{alpha[idx]}. {choice}" for idx, choice in str_choices])
msg = f"# Question: \n{question}\n# Choices: \n{str_choices}"


print(msg)

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

# agent.load_message([{"role": "user", "content": msg}])
# res = agent.get_response(response_format = response_format, temperature = 0, logprobs= True)

input_messages = [{"role": "user", "content": msg}]
top_logprobs = 20
response_raw = agent.openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        response_format = response_format,
        messages=input_messages,
        temperature=1,
        max_tokens=50,
        logprobs = True,
        top_logprobs = top_logprobs
    )

response = res[0]
response_logprobs = res[1]

print("Response:", response)
print("Logprobs:", response_logprobs)
