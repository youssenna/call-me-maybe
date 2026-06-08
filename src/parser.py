from pydantic import (BaseModel, model_validator, FilePath,
                      TypeAdapter, ConfigDict)
from pathlib import Path
import json
from typing import List, Dict, Literal, Any
from sys import exit


class VariableType(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['number', "string", 'boolean']


class FunctionDefinetion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType


class FuctionCallingTest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str

class Parser(BaseModel):
    model_config = ConfigDict(extra='forbid')
    functions_def_file: FilePath
    func_call_test_file: FilePath
    output_file: str
    prompts: List[Any] = []
    functions: List[Any] = []

    @model_validator(mode='after')
    def parse_files(self):
        path = Path(self.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_file).write_text('')

        self.functions.extend(self._parse_functions_def_file(
            self.functions_def_file))
        
        self.prompts.extend(self._parse_prompts_file(
            self.func_call_test_file))
        
        return self
            
    def _parse_functions_def_file(self, functions_def_file: FilePath
                                   ) -> List[Any]:
        with open(functions_def_file, 'r') as f:
            
            data = f.read()
            
            functions_adapter: TypeAdapter = TypeAdapter(
                list[FunctionDefinetion])
            
            valid_functions = functions_adapter.validate_python(
                json.loads(data))
            
            if not valid_functions:
                raise ValueError('can not proccess empty list')
            
            return valid_functions

    def _parse_prompts_file(self, func_call_test_file: FilePath) -> List[Any]:
        with open(func_call_test_file, 'r') as f:
            data = f.read()
            prompt_adapter: TypeAdapter = TypeAdapter(
                list[FuctionCallingTest])
            valid_prompts = prompt_adapter.validate_python(json.loads(data))
            print()
            if not valid_prompts:
                raise ValueError('can not proccess empty list')
            return valid_prompts

    def __str__(self) -> str:
        return f'functions = {self.functions}\n'\
            f'prompts = {self.prompts}'
