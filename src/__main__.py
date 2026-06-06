from sys import argv, exit

from transformers import TokenSpan
from .parser import Parser
from pydantic import ValidationError
import json
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from numpy import array

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
            main_prompt = f2.read() + str(functions)
            print(main_prompt)
        # exit(0)
        for prompt in prompts:
            prompt_dict = prompt.model_dump()
            json_object: str = '{"prompt": '
            main_prompt += f'\n-User request:\n{prompt_dict}\n-JSON object:'\
                           f'\n{json_object}'
            print(main_prompt)
            
            # exit(0)
            tokens = model.encode(main_prompt)
            print(tokens.tolist()[0])
            tokens = tokens.tolist()[0]
            
            print('logit name for prompt: ')
            for x in range(50):
                logits = model.get_logits_from_input_ids(tokens)
                max_logit = max(logits)
                index_logit = logits.index(max_logit)
                logit_name = vocab_dict_miror[index_logit]
            # string = model.decode(logits)
                # print(content)
                tokens.append(index_logit)
                print(logit_name)
    except ValidationError as e:
        for err in e.errors():
            print(err['msg'])