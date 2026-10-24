.PHONY: help install install-dev test test-cov lint format run-api run-frontend \
        train mlflow-ui docker-build docker-up docker-up-dev docker-down docker-logs \
        docker-test docker-train docker-tools clean \
        db-upgrade db-downgrade db-current db-history db-revision create-admin

help:
	@echo "Credit Risk Intelligence Platform — common tasks"
	@echo ""
	@echo "  make install        Install runtime dependencies"
	@echo "  make install-dev    Install runtime + lint/test tooling"
	@echo "  make test           Run the full pytest suite"
	@echo "  make test-cov       Run tests with a coverage report"
	@echo "  make lint           Run ruff + black --check"
	@echo "  make format         Auto-format with black"
	@echo "  make data           Generate the synthetic 50k-application dataset (skipped if present)"
	@echo "  make train          Generate data if needed, then run the full training pipeline"
	@echo "  make run-api        Run the FastAPI backend locally (uvicorn, reload)"
	@echo "  make run-frontend   Run the Streamlit analyst workbench locally"
	@echo "  make run-web        Run the Next.js web console locally (http://localhost:3000)"
	@echo "  make mlflow-ui      Run a local MLflow tracking server"
	@echo "  make db-upgrade     Apply all pending Alembic migrations (DATABASE_URL from env/.env)"
	@echo "  make db-downgrade   Roll back the most recent Alembic migration"
	@echo "  make db-current     Show the current applied migration revision"
	@echo "  make db-history     Show the full migration history"
	@echo "  make db-revision    Autogenerate a new migration from model changes (msg=\"...\")"
	@echo "  make create-admin   Bootstrap the first privileged (admin) account out-of-band"
	@echo "  make docker-build   Build the api + frontend + ml + mlflow images"
	@echo "  make docker-up      Start the PRODUCTION-style stack (postgres/redis/mlflow/api/frontend)"
	@echo "  make docker-up-dev  Start the DEV stack (bind-mounted source, --reload, auto-merges docker-compose.override.yml)"
	@echo "  make docker-down    Stop the docker compose stack"
	@echo "  make docker-logs    Tail logs from the docker compose stack"
	@echo "  make docker-test    Run the pytest suite inside a container (docker-compose.test.yml)"
	@echo "  make docker-train   Run the training pipeline inside the ml-worker container"
	@echo "  make docker-tools   Start optional admin tools (adminer at :8080)"
	@echo "  make clean          Remove caches, __pycache__, coverage artifacts"

install:
	pip install --no-cache-dir -r requirements.txt

install-dev:
	pip install --no-cache-dir -r requirements-dev.txt
	pip install --no-cache-dir ruff==0.8.4 black==24.10.0

test:
	MLFLOW_ALLOW_FILE_STORE=true python3 -m pytest tests/ -v

test-cov:
	MLFLOW_ALLOW_FILE_STORE=true python3 -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint:
	ruff check .
	black --check --diff .

format:
	ruff check --fix .
	black .

DATA_FILE := data/raw/credit_applications.csv

data: $(DATA_FILE)

$(DATA_FILE):
	python3 -m data.generate_synthetic_data

train: $(DATA_FILE)
	MLFLOW_ALLOW_FILE_STORE=true python3 -m pipelines.run_training --stage all

run-api:
	MLFLOW_ALLOW_FILE_STORE=true uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	streamlit run frontend/app.py

run-web:
	cd web && npm ci && npm run dev

mlflow-ui:
	MLFLOW_ALLOW_FILE_STORE=true mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns

# --- Database migrations (Alembic) ---
# See database/migrations/env.py: the connection URL always comes from
# configs.settings.get_settings().database_url (i.e. the DATABASE_URL env
# var / .env), never from alembic.ini, so these targets migrate whichever
# database the app itself is currently configured to use.
db-upgrade:
	python3 -m alembic upgrade head

db-downgrade:
	python3 -m alembic downgrade -1

db-current:
	python3 -m alembic current

db-history:
	python3 -m alembic history --verbose

db-revision:
	python3 -m alembic revision --autogenerate -m "$(msg)"

create-admin:  ## make create-admin username=alice email=alice@bank.com  (password is prompted)
	python3 -m scripts.create_admin --username $(or $(username),admin) --email $(or $(email),admin@example.com)

docker-build:
	docker compose -f docker-compose.yml build

docker-up:
	./deployment/scripts/deploy_docker_compose.sh

docker-up-dev:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-test:
	docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm tests

docker-train:
	docker compose --profile ml run --rm ml-worker

docker-tools:
	docker compose --profile tools up -d adminer

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
