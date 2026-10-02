PYTHON ?= python
# Make "src" importable no matter how Python is configured
export PYTHONPATH := $(CURDIR)

.PHONY: help install data train predict run test docker-up docker-down clean

help:
	@echo "make install     - pip install requirements"
	@echo "make data        - generate synthetic data"
	@echo "make train       - train and evaluate the model"
	@echo "make predict     - print latest risk per unit"
	@echo "make run         - data + train + launch the web app"
	@echo "make test        - run pytest"
	@echo "make docker-up   - build and run with docker compose"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.generate_data

train:
	$(PYTHON) -m src.train

predict:
	$(PYTHON) -m src.predict

run: data train
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true

test:
	$(PYTHON) -m pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -f data/*.csv models/*.joblib models/*.json models/*.csv
