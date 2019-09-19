train_cross:
	@python3 classifier.py --mode train --strategy 'cross' --data '$(MAIL_DATA)/train' --output 'models/cross_model'

predict_cross:
	@python3 classifier.py --mode predict --strategy 'cross' --data '$(MAIL_DATA)/train' --model 'models/cross_model' --output 'predictions/cross_predicition.csv'

eval_cross:
	@python3 classifier.py --mode eval --predictions 'predictions/cross_predicition.csv' --ground_truth '$(MAIL_DATA)/train/labels.csv'

all_cross: train_cross predict_cross eval_cross


train_split:
	@python3 classifier.py --mode train --strategy 'split' --data '$(MAIL_DATA)/train' --output 'models/split_model'

predict_split:
	@python3 classifier.py --mode predict --strategy 'split' --data '$(MAIL_DATA)/train' --model 'models/split_model' --output 'predictions/split_predicition.csv'

eval_split:
	@python3 classifier.py --mode eval --predictions 'predictions/split_predicition.csv' --ground_truth '$(MAIL_DATA)/train/labels.csv'

all_split: train_split predict_split eval_split


train_raw:
	@python3 classifier.py --mode train --strategy 'none' --data '$(MAIL_DATA)/train' --output 'models/raw_model'

predict_raw:
	@python3 classifier.py --mode predict --strategy 'none' --data '$(MAIL_DATA)/test' --model 'models/raw_model' --output 'predictions/submit_predicition.csv'

all_raw: train_raw predict_raw

setup:
	@mkdir -p models predictions
	@pip3 install -r requirements.txt
	@tput setaf 1
	@tput bold
	@echo "Don't forget to define a MAIL_DATA environment variable pointing to the root folder of the dataset"
	@tput setaf sgr0
