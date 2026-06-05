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
        print(parser)
        model = Small_LLM_Model()
        fonctions = parser.functions
        prompts = parser.prompts
        
        vocab_file = model.get_path_to_vocab_file()
        tokonizer_file = model.get_path_to_tokenizer_file()
        merge_file = model.get_path_to_merges_file()
        with (open(vocab_file, 'r') as f1, open(tokonizer_file, 'r') as f2,
              open(merge_file, 'r') as f3):
            vocab_dict = json.load(f1)
            vocab_dict_miror = {v: k for k, v in vocab_dict.items()}
            # print(vocab_dict_miror)
            print('=' * 50, 'tokonizer file', '=' * 50)
            print(f2.read())
            print('=' * 50, 'merge file', '=' * 50)
            print(f3.read())
        for prompt in prompts:
            prompt_dict = prompt.model_dump()
            print(prompt_dict)
            tokens = model.encode(prompt_dict['prompt'] + 'a and b = ')
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