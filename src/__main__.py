from time import time

from typing import List, Dict

from .parser import FunctionDefinetion, Parser, FuctionCallingTest
from pydantic import ValidationError
import json
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
import argparse
from .function_name import select_function_name
from .find_parameters import find_valid_paramters
start_time = time()


def function_calling(model: Small_LLM_Model, main_prompt: List[int],
                     prompts: List[FuctionCallingTest],
                     functions: List[FunctionDefinetion],
                     vocab_dict_miror: Dict[int, str],
                     vocab_dict: Dict[str, int],
                     output_file: str) -> List[Dict[str, str]]:
    '''
    return a list of json objects that contains the function name and the
    paramters vaues.

    :param model: the model name.
    :type model: Small_LLM_Model
    :param main_prompt: the main prompt that I will use to generate the
    function name and the parameters values.
    :type main_prompt: List[int]
    :param prompts: list of prompts.
    :type prompts: List[FuctionCallingTest]
    :param functions: list of available functions.
    :type functions: List[FunctionDefinetion]
    :param vocab_dict_miror: dictionary that maps the token id to the token
    :type vocab_dict_miror: Dict[int, str]
    :param vocab_dict: dictionary that maps the token to the token id
    :type vocab_dict: Dict[str, int]
    :param output_file: the output file that I will save the final results in
    :type output_file: str
    :return: list of json objects that contains the function name and the
    paramters vaues.
    :rtype: List[Dict[str, str]]
    '''
    final_json_obj: List[Dict[str, str]] = []
    i = 0
    for prompt in prompts:
        prompt_dict = prompt.model_dump()
        print('=' * 100)
        print('current prompt: ', end='')
        print(prompt_dict['prompt'])
        json_str: str = '{"prompt": "' + prompt_dict['prompt']
        json_str += '","name": "'

        prefix_prompt = str(f'{prompt_dict}\n-JSON object:'
                            f'\n{json_str}')

        encoded_current_prompt: List[int] = main_prompt + model\
            .encode(prefix_prompt).squeeze().tolist()

        valid_function: str = select_function_name(model,
                                                   vocab_dict_miror,
                                                   functions,
                                                   encoded_current_prompt)
        prom_part: str = valid_function + '","parameters":{"'
        encoded_current_prompt.\
            extend(model.encode(prom_part).squeeze().tolist())

        valid_params: str = '{"'
        valid_params += find_valid_paramters(
                                             model,
                                             encoded_current_prompt,
                                             functions,
                                             valid_function,
                                             vocab_dict_miror,
                                             vocab_dict)
        json_object = {"prompt": prompt_dict['prompt'],
                       "name": valid_function,
                       "parameters": json.loads(valid_params)}

        # print(final_json_obj)
        final_json_obj.append(json_object)
        print('-' * 100, '\ngoted json object I will save it later in '
              f'output file "{output_file}"\n', '#' * 100, '\n')
        print(json.dumps(json_object, indent=4))
        print('\n', '#' * 100, '\n')
        i += 1
    return final_json_obj


def parse_args() -> argparse.Namespace:
    '''
    parse the command line arguments and return the parsed arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    '''
    error_msge = str('uv run python -m src [--functions_definition'
                     ' <function_definition_file>] [--input'
                     ' <input_file>] [--output <output_file>] or\n'
                     'uv run python -m src [--functions_definition'
                     ' <function_definition_file>] [--input'
                     ' <input_file>] [--output <output_file>] --model'
                     ' Qwen/Qwen2.5-1.5B-Instruct')
    arg_parse = argparse.ArgumentParser('call_me_maybe',
                                        error_msge,
                                        'functions calling grneration tool')
    arg_parse.add_argument('--functions_definition')
    arg_parse.add_argument('--input')
    arg_parse.add_argument('--output')
    arg_parse.add_argument('--model')
    args_ = arg_parse.parse_args()
    if not args_.functions_definition:
        args_.functions_definition = 'data/input/functions_definition.json'
    if not args_.input:
        args_.input = 'data/input/function_calling_tests.json'
    if not args_.output:
        args_.output = 'data/output/function_calling_results.json'
    if not args_.model:
        args_.model = 'Qwen/Qwen3-0.6B'
    return args_


def all_functions(functions: List[FunctionDefinetion]) -> str:
    '''
    this function will return a string that contains all the available
    functions in the functions list in a specific format

    :param functions: list of available functions that I will convert it to
    a string in a specific format
    :type functions: List[FunctionDefinetion]
    :return: string that contains all the available functions in a specific
    format
    :rtype: str
    '''
    all_func_str = ''
    for func in functions:
        all_func_str += str(f'-name of function: {func.name}'
                            ': description of function: {func.description}\n')
        all_func_str += f'  parameters of function {func.name}:\n'
        for param_name, param_obj in func.parameters.items():
            all_func_str += f'  -name of parameter: {param_name}:\n'
            all_func_str += f'   -type of parameter: {param_obj.type}\n'
    return all_func_str


def main() -> None:
    """
    Main function to execute the script.
    """
    args = parse_args()
    output_file: str = args.output
    function_def_file = Path(args.functions_definition)
    func_call_test_file = Path(args.input)
    model_name: str = args.model
    parser = Parser(functions_def_file=function_def_file,
                    func_call_test_file=func_call_test_file,
                    output_file=output_file)
    model = Small_LLM_Model(model_name=model_name)

    functions: List[FunctionDefinetion] = parser.functions
    prompts: List[FuctionCallingTest] = parser.prompts

    vocab_file = model.get_path_to_vocab_file()
    with open(vocab_file, 'r') as f1, open('main_prompt.txt', 'r') as f2:
        vocab_dict = json.load(f1)
        vocab_dict_miror = {v: k for k, v in vocab_dict.items()}
        main_prompt = f2.read() + all_functions(functions)
        main_prompt += '\n-User request:\n'

    encoded_main_prompt: List[int] = model.\
        encode(main_prompt).squeeze().tolist()

    final_json_obj = function_calling(model, encoded_main_prompt, prompts,
                                      functions, vocab_dict_miror,
                                      vocab_dict, parser.output_file)
    with open(parser.output_file, 'w') as f:
        json.dump(final_json_obj, f, indent=4)
    end_time = time()
    print(f'# check the {output_file} for to check the final results.')
    print('@' * 100)
    print("@ proccess time:", f'{(end_time - start_time) / 60:.2f}')
    print('@' * 100)


if __name__ == '__main__':
    try:
        main()

    except ValidationError as e:
        for err in e.errors():
            print(err['msg'])

    except BaseException as e:
        print(str(e))
