from openai import OpenAI, APITimeoutError
import os
import json
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
import time


available_models = {
    'openai' : ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "chatgpt-4o-latest"]
}
model_to_model_id = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "chatgpt-4o": "chatgpt-4o-latest"
}


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
            if self.model not in available_models['openai']:
                raise ValueError("Invalid model: ", self.model)
            
            response_raw = self.openai_client.chat.completions.create(
                model=self.model,
                response_format = response_format,
                messages=input_messages,
                temperature=temperature,
                max_tokens=2048,
                logprobs = logprobs,
                top_logprobs = top_logprobs
            )

            if logprobs:
                response = response_raw.choices[0].message.content
                response_logprobs = response_raw.choices[0].logprobs.content
            
                return response, response_logprobs
            else: # if not logprobs
                response = response_raw.choices[0].message.content
                
                if response_format["type"] in ["json_object", "json_schema"]:
                    try:
                        response = json.loads(response)
                    except Exception as e:
                        print(f"Invalid JSON format from OpenAI. Error: {e}.")
                        print(response)
                        return self.get_response(response_format, temperature, logprobs, debug)    
            
                return response
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
        

def top_norm_prob(logprobs, target_output=None):

    if target_output is None:
        # Compute for all tokens if no target is specified
        list_logprobs = [single_token.logprob for single_token in logprobs]
    else:
        # Match tokens that correspond to `target_output`
        matched = False
        accumulated_text = ""
        list_logprobs = []
        for single_token in logprobs:
            accumulated_text += single_token.token
            list_logprobs.append(single_token.logprob)
            if accumulated_text == target_output[:len(accumulated_text)]:
                if accumulated_text.endswith(target_output):
                    matched = True
                    break  # Stop once we’ve matched `target_output`
            else:
                accumulated_text = ""
                list_logprobs = []

        if not matched:
            raise ValueError(f"Target output not found in logprobs: {target_output}")

    # Compute length-normalized log probability
    norm_logprobs = np.mean(list_logprobs)
    norm_probs = np.exp(norm_logprobs)  # Convert log probability back to probability
    return norm_probs

def timed_api_call(func, args=(), max_retries = 5):
    counter = 0
    timeout = 5
    while True:
        try:
            return func_timeout(timeout, func, args=args)
        except FunctionTimedOut:
            counter += 1
            if counter > max_retries:
                print(f"Reached {counter} timeouts in a row.")
        except KeyboardInterrupt:
            print("Operation cancelled by user.")
            raise KeyboardInterrupt
        except Exception as e:
            print(f"API Error: {e}")
            time.sleep(10)
            pass