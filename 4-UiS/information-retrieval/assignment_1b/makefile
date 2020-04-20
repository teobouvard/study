train:
	@python3 classifier.py --mode train --data data/train_data.pkl --model_path models/model1

benchmark:
	@python3 classifier.py --mode benchmark --model models/model1

eval:
	@python3 classifier.py --mode eval --model models/model1 --data data/test_data.pkl --output predictions/third_try_predictions.csv

setup:
	@mkdir -p data models predictions
	@python3 organize_data.py --input $(MAIL_DATA) --output data