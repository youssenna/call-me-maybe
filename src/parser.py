from pydantic import (BaseModel, model_validator, FilePath,
                      TypeAdapter, ConfigDict, Field)
from pathlib import Path
import json
from typing import List, Dict, Literal, Any, Self


class VariableType(BaseModel):
    '''
    this class is responsible for representing the variable type of the
    parameters and returns of the functions defined in the functions
    definition file and it will have a type attribute that represent the
    variable type of the parameters and returns of the functions defined
    in the functions definition file and it will only accept three types
    which are number, string and boolean
    '''

    model_config = ConfigDict(extra='forbid')
    type: Literal['number', "string", 'boolean', 'integer']


class FunctionDefinetion(BaseModel):
    '''
    this class is responsible for representing the functions defined in the
    functions definition file and it will have name, description, parameters
    and returns attributes that represent the name, description, parameters
    and returns of the function defined in the functions definition file
    '''

    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType


class FuctionCallingTest(BaseModel):
    '''
    this class is responsible for representing the prompts defined in the
    prompts file and it will have a prompt attribute that represent the prompt
    defined in the prompts file
    '''
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)


class Parser(BaseModel):
    '''
    this class is responsible for parsing the functions definition file and
    the prompts file and it will store the parsed data in the functions and
    prompts attributes of the Parser class and also it will create the output
    file if it does not exist and it will clear the content of the output file
    if it already exists'''

    model_config = ConfigDict(extra='forbid')
    functions_def_file: FilePath
    func_call_test_file: FilePath
    output_file: str
    prompts: List[Any] = []
    functions: List[Any] = []

    @model_validator(mode='after')
    def parse_files(self) -> Self:
        '''
        this function will parse the functions definition file and the prompts
        file and it will store the parsed data in the functions and prompts
        attributes of the Parser class and also it will create the output file
        if it does not exist and it will clear the content of the output file
        if it already exists

        :param self: self instance of the Parser class
        :return: self instance of the Parser class
        :rtype: Any
        '''

        path = Path(self.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_file).write_text('')

        self.functions.extend(self._parse_functions_def_file(
                              self.functions_def_file))

        self.prompts.extend(self._parse_prompts_file(
            self.func_call_test_file))

        return self

    def _parse_functions_def_file(self, functions_def_file: FilePath
                                  ) -> List[FunctionDefinetion]:
        '''
        this function will parse the functions definition file and it will
        return a list of FunctionDefinetion objects that represent the
        functions defined in the functions definition file and it will also
        validate the data in the functions definition file using the
        FunctionDefinetion model and if the data is not valid it will raise
        a ValueError with a message 'can not proccess empty list' because we
        can not proccess empty list of functions and also if there is any
        error in the data it will raise a ValidationError with the error
        message

        :param self: self instance of the Parser class
        :param functions_def_file: path to the functions definition file
        :type functions_def_file: FilePath
        :return: list of parsed function definitions
        :rtype: List[FunctionDefinetion]
        '''

        with open(functions_def_file, 'r') as f:

            data = f.read()

            functions_adapter: TypeAdapter = TypeAdapter(
                list[FunctionDefinetion])

            valid_functions: List[FunctionDefinetion] = \
                functions_adapter.validate_python(
                json.loads(data))

            if not valid_functions:
                raise ValueError('can not proccess empty list')

            return valid_functions

    def _parse_prompts_file(self, func_call_test_file: FilePath
                            ) -> List[FuctionCallingTest]:
        '''
        this method will parse the prompts file and it will return a list
        of FuctionCallingTest objects that represent the prompts defined in
        the prompts file and it will also validate the data in the prompts
        file using the FuctionCallingTest model and if the data is not valid
        it will raise a ValidationError with a message 'can not proccess empty
        list'

        :param self: self instance of the Parser class
        :param func_call_test_file: path to the function call test file
        :type func_call_test_file: FilePath
        :return: list of parsed function call tests
        :rtype: List[FuctionCallingTest]
        '''

        with open(func_call_test_file, 'r') as f:
            data = f.read()
            prompt_adapter: TypeAdapter = TypeAdapter(
                list[FuctionCallingTest])
            valid_prompts: List[FuctionCallingTest] = \
                prompt_adapter.validate_python(json.loads(data))
            print()
            if not valid_prompts:
                raise ValueError('can not proccess empty list')
            return valid_prompts

    def __str__(self) -> str:
        '''
        this function will return a string representation of the Parser object

        :param self: self instance of the Parser class
        :return: string representation of the Parser object
        :rtype: str
        '''

        return f'functions = {self.functions}\n'\
            f'prompts = {self.prompts}'
