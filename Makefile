# Usage:
# make build_features	# build features from pipeline

PROJECT_PATH=/local_data/housing-prices
SOURCE_PATH=${PROJECT_PATH}/housing-prices

train_model:
	python ${SOURCE_PATH}/models/train_model.py