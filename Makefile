run:
	clear
	uv run python3 -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/output_file.json 