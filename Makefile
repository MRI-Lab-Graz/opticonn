.PHONY: docs-serve docs-build deps

docs-serve:
	@echo "Serving MkDocs documentation locally..."
	./scripts/docs_serve.sh

docs-build:
	@echo "Building MkDocs static site into site/"
	mkdocs build

deps:
	@echo "Install documentation dependencies into active venv (recommended)"
	python -m pip install --upgrade mkdocs mkdocs-material
