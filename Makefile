.PHONY: dev test lint format build run docker-run docker-down clean health

BACKEND := backend
PORT ?= 8000

## dev: install runtime + dev dependencies into the current Python environment
dev:
	pip install -r $(BACKEND)/requirements-dev.txt
	@if [ ! -f $(BACKEND)/.env ]; then \
		cp $(BACKEND)/.env.example $(BACKEND)/.env; \
		echo "Created $(BACKEND)/.env from .env.example -- fill in TAVILY_API_KEY (required) and GROQ_API_KEY or Ollama."; \
	fi

## test: run the full pytest suite
test:
	cd $(BACKEND) && DB_PATH=:memory: TAVILY_API_KEY=$${TAVILY_API_KEY:-ci-placeholder} pytest tests/ -v

## lint: run Ruff (does not modify files)
lint:
	cd $(BACKEND) && ruff check .

## format: apply Black formatting in place, then Ruff's auto-fixable rules
format:
	cd $(BACKEND) && black .
	cd $(BACKEND) && ruff check --fix .

## build: build the production Docker image (tags git commit if available)
build:
	docker build --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
		-t research-agent:latest .

## run: run the API locally with uvicorn --reload (no Docker)
run: dev
	cd $(BACKEND) && uvicorn main:app --reload --port $(PORT)

## docker-run: run the full stack via docker compose (requires `make build` first, or add --build here)
docker-run:
	docker compose up --build

## docker-down: stop and remove the docker compose stack (keeps volumes / data)
docker-down:
	docker compose down

## health: quick health check against a locally running instance
health:
	@./scripts/healthcheck.sh $(PORT)

## clean: remove Python cache artifacts (does not touch data/, .env, or Docker volumes)
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(BACKEND)/.pytest_cache

help:
	@grep -E '^## ' Makefile | sed 's/## /  make /'
