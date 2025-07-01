from openai import OpenAI, APITimeoutError
import os
import json
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
import time
from dotenv import load_dotenv
load_dotenv()
from json_repair import repair_json


available_models = {
    'openai' : ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "chatgpt-4o-latest",
                 "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"],
    'vllm': ["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
             "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen3-32B", "Qwen/Qwen3-14B",
             "Qwen/Qwen3-8B"],
}
model_to_model_id = { #['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano']
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "chatgpt-4o": "chatgpt-4o-latest",
    "llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "qwen-3-32B": "Qwen/Qwen3-32B",
    "qwen-3-14B": "Qwen/Qwen3-14B",
    "qwen-3-8B": "Qwen/Qwen3-8B",
}

def create_openai_client(llm):
    llm = model_to_model_id[llm] if llm in model_to_model_id else llm
    if llm in available_models['openai']:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        openai_client = OpenAI(api_key=api_key)
        return openai_client
    elif llm in available_models['vllm']:
        openai_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY"
            )
        return openai_client
    else:
        raise ValueError(f"Model {llm} is not supported. Available models: {available_models['openai'] + available_models['vllm']}")

class Agent:
    def __init__(self, openai_client, model = "gpt-4o-mini"):
        self.openai_client = openai_client
        self.system_message = []
        self.chat_history = []
        self.model = model_to_model_id[model]

    # def reset_chat(self):
    #     self.chat_history = []

    # def reset_system_message(self):
    #     self.system_message = []

    def load_system_message(self, system_message):
        if type(system_message) == str:
            self.system_message = [{"role": "system", "content": system_message}]
        else:
            raise ValueError("Invalid system message type. Expected string, got ", type(system_message))

    def load_message(self, messages):
        if type(messages) == list:
            self.chat_history = [{"role": message["role"], "content": message["content"]} for message in messages]
        else:
            raise ValueError("Invalid message type. Expected list, got ", type(messages))

    def get_response(self, response_format = {"type": "text"}, temperature = 1, logprobs = False, debug = False):
        try:

            input_messages = self.system_message + self.chat_history

            top_logprobs = 20 if logprobs else None
            if debug:
                print(input_messages)
            if self.model not in available_models['openai']+ available_models['vllm']:
                raise ValueError("Invalid model: ", self.model)
            response_raw = self.openai_client.chat.completions.create(
                model=self.model,
                response_format = response_format,
                messages=input_messages,
                temperature=temperature,
                max_tokens=50,
                logprobs = logprobs,
                top_logprobs = top_logprobs
            )
            response_text = response_raw.choices[0].message.content
            if response_format["type"] in ["json_object", "json_schema"]:
                try:
                    response_json_text = repair_json(response_text)
                    response_json = json.loads(response_json_text)
                except Exception as e:
                    print(f"Invalid JSON format from OpenAI. Error: {e}.")
                    print(response_text)
                    return self.get_response(response_format, temperature, logprobs, debug)
                
                if response_format["type"] == "json_schema":
                    target_keys = response_format["json_schema"]["schema"]["properties"].keys()
                else:
                    target_keys = ["response"]
                json_sanity = json_sanity_check(response_json, keys=target_keys)

                if not json_sanity:
                    # print(response_text)
                    # print("trimmed response:", response_json_text.strip())
                    # print("original length:", len(response_text))
                    # print(response_json)
                    # print("JSON sanity check failed. Retrying...")
                    # return self.get_response(response_format, temperature, logprobs, debug)
                    response_json = {"response": ""}
                
                response_output = response_json
            else:
                response_output = response_text


            if logprobs:
                response_logprobs = response_raw.choices[0].logprobs.content
            
                return response_output, response_logprobs
            else: # if not logprobs
                return response_output
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except APITimeoutError:
            print("API Timeout Error. Retrying after 10 seconds...")
            time.sleep(10)
            return self.get_response(response_format, temperature, logprobs, debug)

    def get_response_timed(self, response_format = {"type": "text"}, temperature = 1, logprobs = False, debug = False, max_retries = 5):
        return timed_api_call(
            lambda: self.get_response(response_format, temperature, logprobs, debug),
            max_retries=max_retries
        )

def replace_tokenizer_artifacts(text, replacements=None):
    # Default replacements if none provided
    if replacements is None:
        replacements = {
            'Ċ': '\n',   # newline
            'Ġ': ' ',    # space before word (used in GPT-2/LLAMA)
            'ĉ': '\t',   # tab
        }
    
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)
    
    return text

def get_top_norm_prob(logprobs, target_output):

    if target_output is None or target_output == "":
        # Compute for all tokens if no target is specified
        list_logprobs = [single_token.logprob for single_token in logprobs]
    else:
        # Match tokens that correspond to `target_output`
        matched = False
        accumulated_text = ""
        list_logprobs = []
        for token_index, single_token in enumerate(logprobs):
            utf8_token = replace_tokenizer_artifacts(single_token.token)
            accumulated_text += utf8_token
            list_logprobs.append(single_token.logprob)
            if accumulated_text == target_output[:len(accumulated_text)]:
                if accumulated_text.endswith(target_output):
                    matched = True
                    break  # Stop once we’ve matched `target_output`
            else:
                accumulated_text = ""
                list_logprobs = []

        if not matched:
            return None, None
            # raise ValueError(f"Target output not found in logprobs: {target_output}")

    # Compute length-normalized log probability
    norm_logprobs = np.mean(list_logprobs)
    norm_probs = np.exp(norm_logprobs)  # Convert log probability back to probability
    return norm_probs, token_index

def top_norm_prob(logprobs, target_output=None):
    norm_probs, token_index = get_top_norm_prob(logprobs, target_output)
    if norm_probs is None:
        norm_probs, token_index = get_top_norm_prob(logprobs, '"'+target_output)
    if norm_probs is None:
        norm_probs, token_index = get_top_norm_prob(logprobs, target_output='"')
    if norm_probs is None:
        raise ValueError(f"Target output {target_output} not found in logprobs: {[single_token.token for single_token in logprobs]}")
    
    return norm_probs, token_index
        

def timed_api_call(func, args=(), max_retries = 5):
    counter = 0
    timeout = 10
    while True:
        try:
            return func_timeout(timeout, func, args=args)
        except FunctionTimedOut:
            counter += 1
            print(f"timeout: {args}")
            # raise APITimeoutError(f"Function timed out after {timeout} seconds. Retry {counter}/{max_retries}.")
            return None, None
            if counter > max_retries:
                print(f"Reached {counter} timeouts in a row.")
        except KeyboardInterrupt:
            print("Operation cancelled by user.")
            raise KeyboardInterrupt
        except Exception as e:
            print(f"API Error: {e}")
            time.sleep(10)
            pass

def json_sanity_check(json_str, keys=["response"]):
    """
    Check if the JSON string is valid and can be parsed.
    If not, attempt to repair it.
    """

    for key in keys:
        if key not in json_str:
            # print(f"Key '{key}' not found in JSON string.")
            return False
        
    else:
        return True
    