run: ## Runs application
	python -m src/facedetect

lint-ruff: ## check lint errors with ruff
	ruff check

format: ## format with Ruff
	ruff format

build: ## build application
	uv build

publish:
	uv publish