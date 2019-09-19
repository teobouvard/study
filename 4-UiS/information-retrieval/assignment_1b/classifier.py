import argparse
import os
import pickle
import re
import email

import numpy as np
import datetime
from dateutil.parser import parse
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

DEV = True
VERBOSE = True

class EmailClassifier():

	def fetch_data(self, data_path, test_size):
		print("Fetching data from {}".format(data_path))

		with open(data_path, 'rb') as f:
			emails = pickle.load(f)

			if DEV:
				self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(emails.data, emails.target, test_size=0.2, random_state=42)
				self.filenames = shuffle(emails.filenames, random_state=42)
			else:
				self.train_X, self.test_X, self.train_Y, self.test_Y = emails.data, None, emails.target, None
				self.filenames = emails.filenames


	def train(self):
		print("Training model on {} train instances".format(len(self.train_X)))

		params = [
			{'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
			{'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
		]

#		model = Pipeline([
#			('text_length', FunctionTransformer(self.text_length_extractor, validate=False)),
#			('special_density', FunctionTransformer(self.special_density_extractor, validate=False)),
#			('hour', FunctionTransformer(self.hour_extractor, validate=False)),
#			('scaler', MinMaxScaler()),
#			('vect', HashingVectorizer()),
#			('domain', CountVectorizer(preprocessor=self.sender_preprocessor)),
#			('subject', CountVectorizer(preprocessor=self.subject_preprocessor)),
#        	('vectorizer', CountVectorizer(preprocessor=self.sender_preprocessor)),
#        	('uppercase_frequency',FunctionTransformer(self.uppercase_extractor, validate=False)),
#			('return', FunctionTransformer(self.return_path_extractor, validate=False)),
#			('tfidf', TfidfTransformer()),
#			('classifier', LinearSVC()),
#		], verbose=True)

		model = Pipeline([
            ('features', FeatureUnion([

				('complete', Pipeline([
                    ('complete_vectorizer', HashingVectorizer()),
                    ('complete_tfidf', TfidfTransformer()),
                ], verbose=VERBOSE)),

				('return', FunctionTransformer(self.return_path_extractor, validate=False)),

                ('special_density', FunctionTransformer(self.special_density_extractor, validate=False)),

				('sender_vectorizer', HashingVectorizer(preprocessor=self.sender_preprocessor)),

            ])),

			('classifier', LogisticRegression())
		], verbose=VERBOSE)

		#grid = GridSearchCV(model, params, n_jobs=4)

		self.classifier = model.fit(self.train_X, self.train_Y)

		#print(grid.best_score_)    
		#print(grid.best_params_)

		break_here = True
	
	def return_path_extractor(self, X):
		return_path = []
		for item in X:
			mail = email.message_from_string(item)
			return_path.append(1 if 'Return-Path' in mail else 0)
		return np.array(return_path).reshape(-1, 1)

	def subject_preprocessor(self, text):
		mail = email.message_from_string(text)
		subject = mail['Subject']
		return subject.lower() if subject is not None else "NONE"
	
	def sender_preprocessor(self, text):
		mail = email.message_from_string(text)
		sender = mail['From']
		sender_domain = sender[sender.find('@')+1:] if sender is not None else None
		return sender_domain.lower() if sender_domain is not None else "NONE"

	def hour_extractor(self, X):
		hours = []
		for item in X:
			mail = email.message_from_string(item)
			try:
				date = parse(mail['Date'], ignoretz=True).time() if mail['Date'] is not None else datetime.time(0, 0, 0)
			except ValueError:
				date = datetime.time(0, 0, 0)
			delta = datetime.timedelta(hours=date.hour, minutes=date.minute, seconds=date.second).total_seconds()
			hours.append(delta) if delta is not None else 0
		return np.array(hours).reshape(-1, 1)

	def special_density_extractor(self, X):
		densities = []
		for mail in X:
			special = re.sub(r'[^!#?<>$%+=()/&]+' ,'', mail) #<>/?!#*
			special_density = len(special)/len(mail)
			densities.append(special_density)
		return np.array(densities).reshape(-1, 1)

	def text_length_extractor(self, X):
		return np.array([len(mail) for mail in X]).reshape(-1, 1)
	
	def uppercase_extractor(self, X):
		count = []
		for mail in X:
			uppercase_words = [word for word in mail.split(" ") if word.isupper()]
			count.append(len(uppercase_words)/len(mail))
		return np.array(count).reshape(-1, 1)

	def benchmark(self):
		if self.test_X is None:
			print("No test data available, are you in DEV mode ?")
			exit()

		print("Making predictions on {} test instances".format(len(self.test_X)))
		predictions = self.classifier.predict(self.test_X)
		tn, fp, fn, tp = confusion_matrix(predictions, self.test_Y).ravel()

		#scores = cross_val_score(self.classifier, self.train_X, self.train_Y, cv=5, n_jobs=-1)

		print("Evaluation results:")
		print("-------------------")
		print("Accuracy:            {:05.3f}".format((tp + tn) / (tp + tn + fp + fn)))
		print("Precision:           {:05.3f}".format(tp / (tp + fp)))
		print("False positive rate: {:05.3f}".format(fp / (fp + tn)))
		print("-------------------")
		#print("CV scores:           {}".format(scores)) #{:05.3f}".format(scores))

		with open('error_checking.txt', 'w') as f:
			for (filename,(pred, actual)) in zip(self.filenames,zip(predictions, self.test_Y)):
				if pred != actual:
					if pred:
						f.write("{} was predicted spam but is ham\n".format(filename))
					else:
						f.write("{} was predicted ham but is spam\n".format(filename))
	
	def eval(self, output_file):
		print("Making predictions on {} instances".format(len(self.train_X)))
		predictions = self.classifier.predict(self.train_X)

		print("Writing predictions to {}".format(output_file))
		with open(output_file, 'w') as f:
			f.write("Id,Label\n")
			for filename, pred in zip(self.filenames, predictions):
				f.write("{},{}\n".format(filename.split("data/")[1], "spam" if pred else "ham"))
	
	def save(self, model_path):
		print("Saving model to {}".format(model_path))

		with open(model_path, 'wb') as f:	
			pickle.dump(self.__dict__, f)			

	def load(self, model_path):
		print("Loading model from {}".format(model_path))

		with open(model_path, 'rb') as f:
			self.__dict__.update(pickle.load(f))



def argument_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument("--data", help="Data pickle generated by the 'organize_data' script.")
	parser.add_argument(
		"--mode",
		choices=["train", "benchmark", "eval"],
		help="Run program in a specific mode.",
	)
	parser.add_argument(
		"--model_path",
		help="Path to the model to save or to load, depending on the mode.",
	)
	parser.add_argument(
		"--output",
		help="Write predictions to a file",
	)

	return parser

if __name__ == "__main__":

	# parse arguments
	parser = argument_parser()
	args = parser.parse_args()

	mode = str(args.mode)
	data_path = str(args.data)
	model_path = str(args.model_path)
	output_file = str(args.output)
	
	# check arguments validity
	if mode == "train" or args.mode == "eval":
		if not os.path.isfile(data_path):
			raise argparse.ArgumentTypeError("Data path must be a valid pickle")
	elif mode == "predict":
		if not os.path.isfile(model_path):
			raise argparse.ArgumentTypeError("Model path must be a valid pickle")
	elif mode == "eval":
		if not os.path.isfile(data_path):
			raise argparse.ArgumentTypeError("Data path must be a valid pickle")
		if not os.path.isfile(model_path):
			raise argparse.ArgumentTypeError("Model path must be a valid pickle")



	# execute specified mode
	if args.mode == "train":

		model = EmailClassifier()
		model.fetch_data(data_path, test_size=0.2 if DEV else 0)
		model.train()
		model.save(model_path)

	elif args.mode == "benchmark":

		model = EmailClassifier()
		model.load(model_path)
		model.benchmark()

	elif args.mode == "eval":

		model = EmailClassifier()
		model.load(model_path)
		model.fetch_data(data_path, test_size=1)
		model.eval(output_file)		
