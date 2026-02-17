.PHONY: install run test test-gpu lint format pre-commit docker-build docker-run docker-build-gpu docker-run-gpu

install:
	uv sync

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

pre-commit:
	uv run pre-commit install

docker-build:
	docker build -t python-service-template .

docker-run:
	docker run -p 8000:8000 python-service-template

test-gpu:
	uv run pytest -m gpu -v

docker-build-gpu:
	docker build -f Dockerfile.gpu -t face-recognition-service:gpu .

docker-run-gpu:
	docker run --gpus all -p 8000:8000 face-recognition-service:gpu
