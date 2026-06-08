from sys import argv, exit
from typing import List, Dict, Any

from transformers import TokenSpan
from .parser import FunctionDefinetion, Parser
from pydantic import ValidationError
import json
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


def get_name_of_valid_function(model: Small_LLM_Model,
                               prompt: str,
                               available_functions: List[FunctionDefinetion],
                               reversed_vocab_dict: Dict[int, str]
                               ) -> str:
    llm_output = ''
    available_fun_names: List[str] = [func.name for func in available_functions]
    while llm_output not in available_fun_names:
        available_fun_names_str: str = ''.join(available_fun_names)
        func_names_tokens = set(model.encode(available_fun_names_str).tolist()[0])
        prompt_to_tokens = model.encode(prompt).tolist()[0]
        logits = np.array(model.get_logits_from_input_ids(prompt_to_tokens))
        constrained_logits = np.full_like(logits, -np.inf)
        valid_tokens_logit_value = logits[list(func_names_tokens)]
        constrained_logits[list(func_names_tokens)] = valid_tokens_logit_value
        llm_output += reversed_vocab_dict[int(constrained_logits.argmax())]
        prompt += llm_output
        # print(llm_output)
    return llm_output


def valid_tokens_for_boolean_param(vocab_dict: Dict[str, int]) -> List[int]:
    return [vocab_dict['true'], vocab_dict['false'], vocab_dict[','],
            vocab_dict['}']]

def valid_tokens_for_number_params(vocab_dict: Dict[int, str]) -> List[int]:
    return [k for k, v in vocab_dict.items() if v.isdigit()
            or v == '-' or v == '.' or v == ',' or v == '}' ]

def find_paramters(model: Small_LLM_Model,
                   prompt: str,
                   available_functions: List[FunctionDefinetion],
                   valid_function: str,
                   reversed_vocab_dict: Dict[int, str],
                   vocab_dict: Dict[str, int]) -> str:
    prompt_len_at_start: int = len(prompt)
    func_object: FunctionDefinetion = list(filter(
        lambda x: x.name == valid_function, available_functions))[0]
    args_name = list(func_object.parameters.keys())
    # her i will store valid characters that's i can accespt in -llm output in
    # case of parameter is number
    digit_tokens: List[int] = valid_tokens_for_number_params(reversed_vocab_dict)
    
    boolean_tokens: List[int] = valid_tokens_for_boolean_param(vocab_dict)
    i = 0
    while i < len(func_object.parameters):
        llm_output: str = ''
        # print('fuck you', len(func_object.parameters))
        # print('#########' * 10)
        # print(prompt)
        # print('#########' * 10)
        prompt += args_name[i] + '": '
        
        if func_object.parameters[args_name[i]].type == 'string':
            # print('<> fuck you: </>')
            prompt += '"'
            while '"' not in llm_output:
                prompt_to_tokens: List[int] = model.encode(prompt).tolist()[0]
                logits: np.ndarray = np.array(model.get_logits_from_input_ids(prompt_to_tokens))
                llm_output = reversed_vocab_dict[int(logits.argmax())].replace('Ġ', ' ')
                if '"' not in llm_output:
                    prompt += llm_output
            prompt += '"'            
        elif (func_object.parameters[args_name[i]].type == 'number' or
             func_object.parameters[args_name[i]].type == 'boolean'):
            # print('<> fuck you: </>')
            while ',' not in llm_output and '}' not in llm_output:
                prompt_to_tokens: List[int] = model.encode(prompt).tolist()[0]
                non_filtred_logits: np.ndarray = np.array(model.get_logits_from_input_ids(prompt_to_tokens))
                filtered_logits: np.ndarray = np.full_like(non_filtred_logits, -np.inf)
                if func_object.parameters[args_name[i]].type == 'number':
                    valid_tokens_logit_value: np.ndarray = non_filtred_logits[digit_tokens]
                    filtered_logits[digit_tokens] = valid_tokens_logit_value
                    llm_output = reversed_vocab_dict[int(filtered_logits.argmax())]
                elif func_object.parameters[args_name[i]].type == 'boolean':
                    valid_tokens_logit_value: np.ndarray = non_filtred_logits[boolean_tokens]
                    filtered_logits[boolean_tokens] = valid_tokens_logit_value
                    llm_output = reversed_vocab_dict[int(filtered_logits.argmax())]
                if ',' not in llm_output and '}' not in llm_output:
                    prompt += llm_output
        if i != len(func_object.parameters)-1:
            prompt += ',"'
        else:
            prompt += '}'
        # print('=' * 100)
        # print('=>: llm output:', llm_output)
        # print('=' * 100)
    
        i += 1
    # print('#' * 100)
    # print('3', prompt)
    # print('#' * 100)

        # if i != len(func_object.parameters):
    prompt += '}'
    return prompt[prompt_len_at_start:]

if __name__ == '__main__':
    if not all([len(argv) == 7, argv[1] == '--functions_definition',
            argv[3] == '--input', argv[5] == '--output']):
        print('error invalid arguments try this syntax:\n'
              'uv run python -m src [--functions_definition'
              ' <function_definition_file>] [--input'
              ' <input_file>] [--output <output_file>]')
        exit(1)
    

    try:
        parser = Parser(functions_def_file=Path(argv[2]),
                        func_call_test_file=Path(argv[4]),
                        output_file=argv[-1])
        # print(parser)
        model = Small_LLM_Model()
        functions = parser.functions
        prompts = parser.prompts
        
        vocab_file = model.get_path_to_vocab_file()
        with open(vocab_file, 'r') as f1, open('main_prompt.txt', 'r') as f2:
            vocab_dict = json.load(f1)
            vocab_dict_miror = {v: k for k, v in vocab_dict.items()}
            main_prompt = f2.read() + str(functions) + f'\n-User request:\n'
        # exit(0)
        for prompt in prompts:
            prompt_dict = prompt.model_dump()
            json_object: str = '{"prompt": "' + prompt_dict['prompt'] + '","name": "'
            current_prompt = main_prompt + f'{prompt_dict}\n-JSON object:'\
                           f'\n{json_object}'
            valid_function: str = get_name_of_valid_function(model,
                                                         current_prompt,
                                                         functions,
                                                         vocab_dict_miror)
            json_object += valid_function + '","parameters":{"'
            current_prompt += valid_function + '","parameters":{"'
            valid_params: str = find_paramters(model, current_prompt, functions, valid_function, vocab_dict_miror, vocab_dict)
            json_object += valid_params
            # current_prompt += valid_params
            print('current prompt\n', '#' * 100, '\n',current_prompt)
            print('json\n', '#' * 100, '\n',json_object, '\n', '#' * 100, '\n')
            
    except ValidationError as e:
        for err in e.errors():
            print(err['msg'])