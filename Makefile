install:
	uv sync



run:
	clear
	uv run python3 -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/output_file.json
bonus:
	clear
	uv run python3 -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/output_file.json --model Qwen/Qwen2.5-1.5B-Instruct


debug:
	clear
	uv run python3 -m pdb -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/output_file.json


debug_bonus:
	clear
	uv run python3 -m pdb -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json --output data/output/output_file.json --model Qwen/Qwen2.5-1.5B-Instruct


clean:
	rm -rf __pycache__ .mypy_cache cache .venv data/output */__pycache__

lint:
	uv run flake8 .  --exclude='llm_sdk cache .venv'
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --disable-error-code=list-item --exclude 'cache|.venv|llm_sdk'
