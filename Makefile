.PHONY: setup download train evaluate results all clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install package and dev dependencies
	python -m pip install -e '.[dev]'

download: setup ## Download dataset and pre-pull embedding model
	python -m er.download
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

train: download ## Run blocking + feature extraction + model training + calibration
	python -m er --config configs/amazon_google.yaml --step train

evaluate: train ## Evaluate model and generate results
	python -m er --config configs/amazon_google.yaml --step evaluate

results: evaluate ## Alias for evaluate

all: evaluate ## Run the full pipeline end-to-end

clean: ## Remove generated artifacts and results
	rm -rf artifacts/ results/*.png results/metrics.json
	@echo "Cleaned artifacts and results."
