
SHELL := /bin/bash

MODEL_FILE = "model.yml"
CLEAR_LOG = false
LOG_LEVEL = INFO

setup: create_venv
	@echo "Activate the venv with: \`source ./llm_validator/bin/activate\`"

create_venv:
	python3 -m venv ./llm_validator ;\
	source ./llm_validator/bin/activate ;\
	pip3 install -q -r requirements.txt

.PHONY: clean
clean:
	rm -f logger.out ;\
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: destroy
destroy: clean
	rm -rf ./llm_validator

from_yml:
	@if [ "$(CLEAR_LOG)" = "true" ] || [ "$(CLEAR_LOG)" = "1" ]; then \
        rm logger.out ;\
        echo "logger.out deleted" ;\
    fi
	LOG_LEVEL=$(LOG_LEVEL) python3 ./main.py --from_yml $(MODEL_FILE)
