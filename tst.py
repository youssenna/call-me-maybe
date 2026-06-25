import json


with open('data/output/output_file.json', 'r') as f:
    functions_definition = json.load(f)
    print(functions_definition)