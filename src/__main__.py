from sys import argv, exit
from typing import List, Dict

from .parser import FunctionDefinetion, Parser, FuctionCallingTest
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
    '''
    this function will get the name of the valid function that I will call
    it in the end and I will get it by sending the prompt to the model and
    get the output and compare it with the available functions name and I
    will do that until I get the valid function name

    :param model: model to use it in encoding and getting logits from input
        ids

    :type model: Small_LLM_Model
    :param prompt: prompt that I will send to the model to get the name of
        valid function

    :type prompt: str
    :param available_functions: list of available functions that I will
        compare the model output with it to get the valid function name

    :type available_functions: List[FunctionDefinetion]
    :param reversed_vocab_dict: dictionary mapping token IDs to their
    string representations

    :type reversed_vocab_dict: Dict[int, str]
    :return: name of the valid function
    :rtype: str
    '''
    llm_output = ''
    available_fun_names: List[str] = \
        [func.name for func in available_functions]

    while llm_output not in available_fun_names:
        available_fun_names_str: str = ''.join(available_fun_names)

        func_names_tokens = \
            set(model.encode(available_fun_names_str).tolist()[0])

        prompt_to_tokens = model.encode(prompt).tolist()[0]
        logits = np.array(model.get_logits_from_input_ids(prompt_to_tokens))
        constrained_logits = np.full_like(logits, -np.inf)
        valid_tokens_logit_value = logits[list(func_names_tokens)]
        constrained_logits[list(func_names_tokens)] = valid_tokens_logit_value
        llm_output += reversed_vocab_dict[int(constrained_logits.argmax())]
        prompt += llm_output
    return llm_output


def valid_tokens_for_boolean_param(vocab_dict: Dict[str, int]) -> List[int]:
    '''
    this function will return the valid tokens that I can accept in llm
    output in case of parameter is boolean because I have only two values for
    boolean true and false so I will return the tokens of true and false and
    also I will return the tokens of ',' and '}' because they are the only
    valid tokens that I can accept after the boolean value because I will
    have only two cases after the boolean value either I will have ',' if
    there are more parameters or I will have '}' if the boolean parameter is
    the last parameter in the parameters list

    :param vocab_dict: dictionary mapping string representations to their token
    IDs

    :type vocab_dict: Dict[str, int]
    :return: list of valid token IDs for boolean parameters
    :rtype: List[int]
    '''
    return [vocab_dict['true'], vocab_dict['false'], vocab_dict[','],
            vocab_dict['}']]


def valid_tokens_for_number_params(vocab_dict: Dict[int, str]) -> List[int]:
    '''
    this function will return the valid tokens that I can accept in llm
    output in case of parameter is number because I have many values for
    number and I will return the tokens of digits from 0 to 9 and also I
    will return the tokens of ',' and '}' because they are the only valid
    tokens that I can accept after the number value because I will have
    only two cases after the number value either I will have ',' if there
    are more parameters or I will have '}' if the number parameter is the
    last parameter in the parameters list

    :param vocab_dict: dictionary mapping token IDs to their string
    representations
    :type vocab_dict: Dict[int, str]
    :return: list of valid token IDs for number parameters
    :rtype: List[int]
    '''
    return [k for k, v in vocab_dict.items() if v.isdigit()
            or v == '-' or v == '.' or v == ',' or v == '}']


def find_len_of_last_digit(digits_llm_output: str, llm_output: str) -> int:
    '''
    this function will find the length of the last digit in the llm output
    because I want to add ',' after the number if the number is too long to
    prevent the model from generating very long numbers that may cause
    overflow

    :param digits_llm_output: string that contains only the digits that the
    model generated for the number parameter
    :type digits_llm_output: str
    :param llm_output: the last output that the model generated for the number
    parameter
    :type llm_output: str
    :return: length of the last digit in the llm output
    :rtype: int
    '''
    i = 0
    if len(llm_output) == 1 and llm_output.isdigit():
        for c in digits_llm_output:
            if c == llm_output:
                i += 1
            else:
                i = 0
    return i


def find_paramters(model: Small_LLM_Model,
                   prompt: str,
                   available_functions: List[FunctionDefinetion],
                   valid_function: str,
                   reversed_vocab_dict: Dict[int, str],
                   vocab_dict: Dict[str, int]) -> str:
    '''
    this function will find the parameters of the valid function that I will
    call in the end and I will get it by sending the prompt to the model and
    get the output and compare it with the valid tokens for each parameter
    type and I will do that until I get all the parameters of the valid
    function and I will return the parameters in json format as a string
    to be added in the final json object that I will save it in the output
    file

    :param model: model to use it in encoding and getting logits from input
    ids
    :type model: Small_LLM_Model

    :param prompt: prompt that I will send to the model to get the parameters
    of the valid function
    :type prompt: str

    :param available_functions: list of available functions that I will
    compare the model output with it to get the valid function name and also
    to get the parameters of the valid function
    :type available_functions: List[FunctionDefinetion]

    :param valid_function: name of the valid function that I will call in the
    end to get its parameters
    :type valid_function: str

    :param reversed_vocab_dict: dictionary mapping token IDs to their string
    representations
    :type reversed_vocab_dict: Dict[int, str]

    :param vocab_dict: dictionary mapping string representations to token IDs
    :type vocab_dict: Dict[str, int]

    :return: parameters of the valid function in JSON format
    :rtype: str
    '''
    prompt_len_at_start: int = len(prompt)
    func_object: FunctionDefinetion = list(filter(
        lambda x: x.name == valid_function, available_functions))[0]
    args_name = list(func_object.parameters.keys())
    # her i will store valid tokens that's i can accept in -llm output in
    # case of parameter is number
    digit_tokens: List[int] = \
        valid_tokens_for_number_params(reversed_vocab_dict)
    # her i will store valid tokens that's i can accept in -llm output in
    # case of parameter is boolean
    boolean_tokens: List[int] = valid_tokens_for_boolean_param(vocab_dict)
    i = 0
    while i < len(func_object.parameters):
        llm_output: str = ''
        prompt += args_name[i] + '": '

        if func_object.parameters[args_name[i]].type == 'string':
            prompt += '"'
            while '"' not in llm_output:
                prompt_to_tokens: List[int] = model.encode(prompt).tolist()[0]

                logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(prompt_to_tokens))

                llm_output = reversed_vocab_dict[int(logits.argmax())].\
                    replace('Ġ', ' ')

                if '"' not in llm_output:
                    prompt += llm_output
            prompt += '"'
        elif (func_object.parameters[args_name[i]].type == 'number' or
              func_object.parameters[args_name[i]].type == 'boolean'):
            digits_llm_output: str = ''
            while ',' not in llm_output and '}' not in llm_output:
                prompt_to_tokens = model.encode(prompt).tolist()[0]

                non_filtred_logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(prompt_to_tokens))

                filtered_logits: np.ndarray = np.\
                    full_like(non_filtred_logits, -np.inf)

                if func_object.parameters[args_name[i]].type == 'number':

                    valid_tokens_logit_value: np.ndarray = \
                        non_filtred_logits[digit_tokens]

                    filtered_logits[digit_tokens] = valid_tokens_logit_value

                    llm_output = \
                        reversed_vocab_dict[int(filtered_logits.argmax())]

                    digits_llm_output += llm_output

                    len_last_digits: int = \
                        find_len_of_last_digit(digits_llm_output, llm_output)
                    if len_last_digits > 20:
                        llm_output += ','
                elif func_object.parameters[args_name[i]].type == 'boolean':
                    valid_tokens_logit_value =\
                        non_filtred_logits[boolean_tokens]

                    filtered_logits[boolean_tokens] = valid_tokens_logit_value
                    llm_output = \
                        reversed_vocab_dict[int(filtered_logits.argmax())]

                if ',' not in llm_output and '}' not in llm_output:
                    prompt += llm_output
        if i != len(func_object.parameters)-1:
            prompt += ',"'
        else:
            prompt += '}'

        i += 1
    prompt += '}'
    return prompt[prompt_len_at_start:]


def function_calling(model: Small_LLM_Model, main_prompt: str,
                     prompts: List[FuctionCallingTest],
                     functions: List[FunctionDefinetion],
                     vocab_dict_miror: Dict[int, str],
                     vocab_dict: Dict[str, int]) -> str:
    '''
    this function will be responsible for the main logic of the program and it
    will call the get_name_of_valid_function and find_parameters functions to
    get the name of the valid function and its parameters for each prompt in
    the prompts list and it will return the final json object that I will save
    in the output file and it will also print the json object in a readable
    format in the console

    :param model: model to use it in encoding and getting logits from input
    ids
    :type model: Small_LLM_Model

    :param main_prompt: main prompt that I will add the current prompt and the
    json object that I will get from the model to it to get the name of the
    valid function and its parameters
    :type main_prompt: str

    :param prompts: list of prompts that I will send to the model to get the
    name of the valid function and its parameters for each prompt
    :type prompts: List[FuctionCallingTest]

    :param functions: list of available functions that I will compare the
    model output with it to get the valid function name and also to get the
    parameters of the valid function
    :type functions: List[FunctionDefinetion]

    :param vocab_dict_miror: dictionary mapping token IDs to their string
    representations
    :type vocab_dict_miror: Dict[int, str]

    :param vocab_dict: dictionary mapping string representations to token IDs
    :type vocab_dict: Dict[str, int]
    :return: final json object that I will save it in the output file
    :rtype: str
    '''
    final_json_obj: str = '['
    i = 0
    for prompt in prompts:
        prompt_dict = prompt.model_dump()
        prompt_dict['prompt'] = prompt_dict['prompt'].replace('\"', '\'')
        print('=' * 100)
        print('current prompt: ', end='')
        print(prompt_dict['prompt'])
        json_object: str = '{"prompt": "' + prompt_dict['prompt'] \
                                          + '","name": "'

        current_prompt = main_prompt + f'{prompt_dict}\n-JSON object:'\
                                       f'\n{json_object}'
        valid_function: str = get_name_of_valid_function(model,
                                                         current_prompt,
                                                         functions,
                                                         vocab_dict_miror)
        json_object += valid_function + '","parameters":{"'
        current_prompt += valid_function + '","parameters":{"'
        valid_params: str = find_paramters(
                                            model,
                                            current_prompt,
                                            functions,
                                            valid_function,
                                            vocab_dict_miror,
                                            vocab_dict)

        json_object += valid_params
        final_json_obj += json_object
        if i != len(prompts) - 1:
            final_json_obj += ','
        print('-' * 100, '\ngoted json object I will save it later in '
              f'output file "{argv[6]}"\n', '#' * 100, '\n')
        j_obj = json.loads(json_object)
        print(json.dumps(j_obj, indent=4))
        print('\n', '#' * 100, '\n')
        i += 1
    final_json_obj += ']'
    return final_json_obj


if __name__ == '__main__':
    if not all([len(argv) == 7 or len(argv) == 9,
                argv[1] == '--functions_definition',
                argv[3] == '--input', argv[5] == '--output']):
        print('error invalid arguments try this syntax:\n'
              'uv run python -m src [--functions_definition'
              ' <function_definition_file>] [--input'
              ' <input_file>] [--output <output_file>] or\n'
              'uv run python -m src [--functions_definition'
              ' <function_definition_file>] [--input'
              ' <input_file>] [--output <output_file>] --model'
              ' Qwen/Qwen2.5-1.5B-Instruct')
        exit(1)

    if len(argv) == 9:
        if not all([argv[7] == '--model',
                    argv[-1] == 'Qwen/Qwen2.5-1.5B-Instruct']):
            print('error invalid arguments try this syntax:\n'
                  'uv run python -m src [--functions_definition'
                  ' <function_definition_file>] [--input'
                  ' <input_file>] [--output <output_file>] or\n'
                  'uv run python -m src [--functions_definition'
                  ' <function_definition_file>] [--input'
                  ' <input_file>] [--output <output_file>] --model'
                  ' Qwen/Qwen2.5-1.5B-Instruct')
            exit(1)
        model_name: str = argv[-1]

    try:
        parser = Parser(functions_def_file=Path(argv[2]),
                        func_call_test_file=Path(argv[4]),
                        output_file=argv[6])
        # model = Small_LLM_Model(model_name=model_name)
        if len(argv) == 9:
            print(' '*40, 'Current model')
            print('#'*100)
            print('#', ' '*40, model_name)
            print('#'*100)
            model = Small_LLM_Model(model_name=model_name)
        else:
            print(' '*40, 'Current model')
            print('#'*100)
            print('#', ' '*40, 'Qwen/Qwen3-0.6B')
            print('#'*100)
            model = Small_LLM_Model()

        functions: List[FunctionDefinetion] = parser.functions
        prompts: List[FuctionCallingTest] = parser.prompts

        vocab_file = model.get_path_to_vocab_file()
        with open(vocab_file, 'r') as f1, open('main_prompt.txt', 'r') as f2:
            vocab_dict = json.load(f1)
            vocab_dict_miror = {v: k for k, v in vocab_dict.items()}
            main_prompt = f2.read() + str(functions) + '\n-User request:\n'
        final_json_obj = function_calling(model, main_prompt, prompts,
                                          functions, vocab_dict_miror,
                                          vocab_dict)
        with open(argv[6], 'w') as f:
            obj = json.loads(final_json_obj)
            json.dump(obj, f, indent=4)

    except ValidationError as e:
        for err in e.errors():
            print(err['msg'])

    except Exception as e:
        print(str(e))
