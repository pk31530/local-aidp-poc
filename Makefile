.PHONY: bootstrap start stop health seed train demo api dashboard consumer test reset

bootstrap:
	./scripts/bootstrap.sh

start:
	./scripts/start.sh

stop:
	./scripts/stop.sh

health:
	./scripts/healthcheck.sh

seed:
	./scripts/seed_data.sh

train:
	./scripts/train_model.sh

demo:
	./scripts/run_stream.sh --rate 10 --duration 30

api:
	./scripts/run_api.sh

dashboard:
	./scripts/run_dashboard.sh

consumer:
	./scripts/run_consumer.sh

test:
	. .venv/bin/activate && pytest

reset:
	./scripts/reset_demo.sh
