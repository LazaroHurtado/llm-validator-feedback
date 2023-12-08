
SHELL := /bin/bash

setup: create_venv
	@echo "Activate the venv with: \`source ./llm_validator/bin/activate\`"

create_venv:
	python3 -m venv ./llm_validator ;\
	source ./llm_validator/bin/activate ;\
	pip3 install -q -r requirements.txt

.PHONY: clean
clean:
	rm -rf ./llm_validator

from_yml:
	python3 ./main.py --from_yml "model.yml"