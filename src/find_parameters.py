import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List, Dict
from .parser import FunctionDefinetion


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


def find_valid_paramters(model: Small_LLM_Model,
                   encoded_prompt: List[int],
                   available_functions: List[FunctionDefinetion],
                   valid_function: str,
                   reversed_vocab_dict: Dict[int, str],
                   vocab_dict: Dict[str, int]) -> str:
    '''
    this function will find the parameters of the valid function that I will
    call in the end

    :param model: model to use it in encoding and getting logits from input
    ids
    :type model: Small_LLM_Model

    :param prompt: prompt that I will send to the model to get the parameters
    of the valid function
    :type prompt: str

    :param available_functions: list of available functions that I will
    compare the model output with it to get the valid function name
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
    prompt_len_at_start: int = len(encoded_prompt)
    func_object: FunctionDefinetion = list(filter(
        lambda x: x.name == valid_function, available_functions))[0]
    args_name = list(func_object.parameters.keys())
    
    # print(args_name)
    # exit()
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
        print(encoded_prompt)
        encoded_prompt.extend(model.encode(args_name[i] + '": ').squeeze().tolist())
        
        print()
        if func_object.parameters[args_name[i]].type == 'string':
            encoded_prompt += [model.encode('"').squeeze().tolist()]
            while '"' not in llm_output:
                # print(llm_output)
                # prompt_to_tokens: List[int] = model.encode(prompt).tolist()[0]

                logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(encoded_prompt))

                llm_output = reversed_vocab_dict[int(logits.argmax())].\
                    replace('Ġ', ' ')

                if '"' not in llm_output:
                    encoded_prompt += [model.encode(llm_output).squeeze().tolist()]
                else:
                    encoded_prompt += [model.encode(llm_output[:llm_output.index('"')]).squeeze().tolist()]
            encoded_prompt += [model.encode('"').squeeze().tolist()]
        elif (func_object.parameters[args_name[i]].type == 'number' or
              func_object.parameters[args_name[i]].type == 'boolean' or
              func_object.parameters[args_name[i]].type == 'integer'):
            digits_llm_output: str = ''
            while ',' not in llm_output and '}' not in llm_output:
                print(llm_output)
                # prompt_to_tokens = model.encode(prompt).tolist()[0]
                print(encoded_prompt)
                non_filtred_logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(encoded_prompt))

                filtered_logits: np.ndarray = np.\
                    full_like(non_filtred_logits, -np.inf)

                if (func_object.parameters[args_name[i]].type == 'number' or
                   func_object.parameters[args_name[i]].type == 'integer'):

                    valid_tokens_logit_value: np.ndarray = \
                        non_filtred_logits[digit_tokens]

                    filtered_logits[digit_tokens] = valid_tokens_logit_value

                    llm_output = \
                        reversed_vocab_dict[int(filtered_logits.argmax())]

                    digits_llm_output += llm_output
                    # print(llm_output)

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
                    encoded_prompt += [model.encode(llm_output).squeeze().tolist()]
        if i != len(func_object.parameters)-1:
            encoded_prompt += [model.encode(',"').squeeze().tolist()]
        else:
            encoded_prompt += [model.encode('}').squeeze().tolist()]
        i += 1
    return model.decode(encoded_prompt[prompt_len_at_start:])

