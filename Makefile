UV ?= uv
# Retrieval config whose chunks + index `make data` builds (the current default).
CONFIG ?= configs/pages-context.yaml

.PHONY: install format lint typecheck test test-slow check download-financebench data

install:  ## Create .venv and install all dependencies
	$(UV) sync

format:  ## Auto-format and auto-fix lint issues
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

test:  ## Fast, offline unit tests
	$(UV) run pytest --cov=rag --cov-report=term-missing

test-slow:  ## Tests that need network / real models
	$(UV) run pytest -m slow

check: lint typecheck test  ## Everything CI runs

download-financebench:  ## Metadata + the 84 PDFs referenced by the open-source questions
	$(UV) run python scripts/download_financebench.py --scope questions

data: download-financebench  ## Download + parse + chunk + embed for CONFIG (skips steps already done)
	$(UV) run python scripts/parse_documents.py --config $(CONFIG)
	$(UV) run python scripts/chunk_documents.py --config $(CONFIG)
	$(UV) run python scripts/build_index.py --config $(CONFIG)
