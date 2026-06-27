import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List, Dict
from .parser import FunctionDefinetion


def valid_tokens_for_boolean_param(vocab_dict: Dict[str, int]) -> List[int]:
    '''
    return the valid tokens that I can accept in llm output in case of
    parameter is boolean.
    :param vocab_dict: dictionary mapping string representations to their
    token IDs
    :type vocab_dict: Dict[str, int]
    :return: list of valid token IDs for boolean parameters
    :rtype: List[int]
    '''
    return [vocab_dict['true'], vocab_dict['false'], vocab_dict[','],
            vocab_dict['}']]


def valid_tokens_for_number_params(vocab_dict: Dict[int, str]) -> List[int]:
    '''
    return the valid tokens that I can accept in llm output in case of
    parameter is number.

    :param vocab_dict: dictionary mapping token IDs to their string
    representations
    :type vocab_dict: Dict[int, str]
    :return: list of valid token IDs for number parameters
    :rtype: List[int]
    '''
    return [k for k, v in vocab_dict.items() if v.isdigit()
            or v == '-' or v == '.' or v == ',' or v == '}']


def valid_tokens_for_integer_params(vocab_dict: Dict[int, str]) -> List[int]:
    '''
    return the valid tokens that I can accept in llm output in case of
    parameter is integer.

    :param vocab_dict: dictionary mapping token IDs to their string
    representations
    :type vocab_dict: Dict[int, str]
    :return: list of valid token IDs for number parameters
    :rtype: List[int]
    '''
    return [k for k, v in vocab_dict.items() if v.isdigit()
            or v == '-' or v == ',' or v == '}']


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

    :param model: the small LLM model that I will use to generate the
    parameters of the valid function
    :type model: Small_LLM_Model
    :param encoded_prompt: the encoded prompt that I will use to generate the
    parameters of the valid function
    :type encoded_prompt: List[int]
    :param available_functions: list of available functions that I will use to
    find the parameters of the valid function
    :type available_functions: List[FunctionDefinetion]
    :param valid_function: the name of the valid function that I will find its
    parameters
    :type valid_function: str
    :param reversed_vocab_dict: dictionary mapping token IDs to their string
    representations
    :type reversed_vocab_dict: Dict[int, str]
    :param vocab_dict: dictionary mapping string representations to their
    token IDs
    :type vocab_dict: Dict[str, int]
    :return: the parameters of the valid function in string format
    :rtype: str
    '''
    prompt_len_at_start: int = len(encoded_prompt)
    func_object: FunctionDefinetion = list(filter(
        lambda x: x.name == valid_function, available_functions))[0]
    args_name = list(func_object.parameters.keys())

    # her i will store valid tokens that's i can accept in -llm output in
    # case of parameter is number
    number_tokens: List[int] = \
        valid_tokens_for_number_params(reversed_vocab_dict)
    integer_tokens: List[int] = \
        valid_tokens_for_integer_params(reversed_vocab_dict)
    # her i will store valid tokens that's i can accept in -llm output in
    # case of parameter is boolean
    boolean_tokens: List[int] = valid_tokens_for_boolean_param(vocab_dict)
    i = 0
    while i < len(func_object.parameters):
        llm_output: str = ''
        encoded_prompt.\
            extend(model.encode(args_name[i] + '": ').squeeze().tolist())

        if func_object.parameters[args_name[i]].type == 'string':
            # print([model.encode('"').squeeze().tolist())
            encoded_prompt += [model.encode('"').squeeze().tolist()]

            while '"' not in llm_output:

                logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(encoded_prompt))

                llm_output = reversed_vocab_dict[int(logits.argmax())].\
                    replace('Ġ', ' ')

                if '"' not in llm_output:
                    enc_out = model.encode(llm_output).squeeze().tolist()
                    encoded_prompt.\
                        extend([enc_out])
                else:
                    remaining_output: str = llm_output[:llm_output.index('"')]
                    if remaining_output:
                        en_rem = model.\
                            encode(remaining_output).squeeze().tolist()

                        encoded_prompt.extend([en_rem])

            encoded_prompt += [model.encode('"').squeeze().tolist()]
        elif (func_object.parameters[args_name[i]].type == 'number' or
              func_object.parameters[args_name[i]].type == 'boolean' or
              func_object.parameters[args_name[i]].type == 'integer'):
            digits_llm_output: str = ''
            while ',' not in llm_output and '}' not in llm_output:
                non_filtred_logits: np.ndarray = np.\
                    array(model.get_logits_from_input_ids(encoded_prompt))

                filtered_logits: np.ndarray = np.\
                    full_like(non_filtred_logits, -np.inf)

                if func_object.parameters[args_name[i]].type == 'number':

                    valid_tokens_logit_value: np.ndarray = \
                        non_filtred_logits[number_tokens]

                    filtered_logits[number_tokens] = valid_tokens_logit_value

                    llm_output = \
                        reversed_vocab_dict[int(filtered_logits.argmax())]

                    digits_llm_output += llm_output

                    len_last_digits: int = \
                        find_len_of_last_digit(digits_llm_output, llm_output)
                    if len_last_digits > 20:
                        llm_output += ','
                elif func_object.parameters[args_name[i]].type == 'integer':
                    valid_tokens_logit_value = \
                        non_filtred_logits[integer_tokens]

                    filtered_logits[integer_tokens] = valid_tokens_logit_value

                    llm_output = \
                        reversed_vocab_dict[int(filtered_logits.argmax())]

                    digits_llm_output += llm_output

                    len_last_digits = \
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
                    en_out = model.encode(llm_output).squeeze().tolist()
                    encoded_prompt.extend([en_out])
        if i != len(func_object.parameters)-1:
            encoded_prompt.extend([model.encode(',"').squeeze().tolist()])
        else:
            encoded_prompt.extend([model.encode('}').squeeze().tolist()])
        i += 1
    return model.decode(encoded_prompt[prompt_len_at_start:])
