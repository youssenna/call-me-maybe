from typing import List, Dict
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import FunctionDefinetion


def select_function_name(model: Small_LLM_Model,
                         reversed_vocab_dict: Dict[int, str],
                         available_functions: List[FunctionDefinetion],
                         encoded_prompt: List[int]) -> str:

    llm_output = ''
    allowed_tokens_list = []
    for func in available_functions:
        tokens = model.encode(func.name).squeeze().tolist()
        allowed_tokens_list.append(tokens if type(tokens) == list else [tokens])
        
    i = 0
    while allowed_tokens_list:
        allowed_tokens = {tokens[0] for tokens in allowed_tokens_list if tokens}
        if not allowed_tokens:
            break
        # prompt_to_tokens = model.encode(prompt).squeeze().tolist()
        logits = np.array(model.get_logits_from_input_ids(encoded_prompt))
        
        constrained_logits = np.full_like(logits, -np.inf)
        valid_tokens_logit_value = logits[list(allowed_tokens)]
        constrained_logits[list(allowed_tokens)] = valid_tokens_logit_value
        llm_output += reversed_vocab_dict[int(constrained_logits.argmax())]
        encoded_prompt += [model.encode(reversed_vocab_dict[int(constrained_logits.argmax())]).squeeze().tolist()]
        allowed_tokens_list = [tokens[1:] for tokens in allowed_tokens_list if tokens and tokens[0] == constrained_logits.argmax()]
    # print(llm_output)

    return llm_output

            
            