COMPOSE ?= docker compose
DAY ?= day_01.csv
CSV ?= /data/$(DAY)

.PHONY: help build train serve up test reload health ready predict logs ps stop cleanup nuke

help:
	@echo "Targets:"
	@echo "  make build                 Build all images"
	@echo "  make train DAY=day_01.csv  Train model with /data/<DAY>"
	@echo "  make serve                 Start API in background"
	@echo "  make up                    Build + start API"
	@echo "  make test                  Run test container"
	@echo "  make reload                Reload model in running API"
	@echo "  make health                Check /healthz"
	@echo "  make ready                 Check /readyz"
	@echo "  make predict               Send sample /predict request"
	@echo "  make logs                  Tail API logs"
	@echo "  make ps                    Show compose services"
	@echo "  make stop                  Stop services"
	@echo "  make cleanup               Stop + remove containers/networks"
	@echo "  make nuke                  cleanup + remove artifacts volume"

build:
	$(COMPOSE) build

train:
	$(COMPOSE) run --rm trainer --csv $(CSV)

serve:
	$(COMPOSE) up -d api

up: build serve

test:
	$(COMPOSE) run --rm --build test

reload:
	curl -sS -X POST http://localhost:8000/model/reload

health:
	curl -sS http://localhost:8000/healthz

ready:
	curl -sS http://localhost:8000/readyz

predict:
	curl -sS -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"age":42,"income":85000,"account_balance":12000,"transactions_last_30d":18,"is_premium":1}'

logs:
	$(COMPOSE) logs -f api

ps:
	$(COMPOSE) ps

stop:
	$(COMPOSE) stop

cleanup:
	$(COMPOSE) down --remove-orphans

nuke:
	$(COMPOSE) down -v --remove-orphans
