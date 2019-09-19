import argparse
import csv
import json
import os
import pickle
import re
import sys
from collections import Counter

from sklearn import metrics
from tqdm import tqdm

LOG_LEVEL = 0  # set to 1 for verbose mode

SPLIT_RATIO = 0.7 # training/validation ratio used in split validation
KFOLDS = 5 # number of folds used in cross validation
WORDLIST_LENGTH = 5000  # limit wordlist size to speed up training

SIGNIFICANT_WORDS = r'\w+' #r'[a-z]\w+[a-z]' # what does a 'word' look like ?
SUBJECT_WORDS = '(?<=subject:).*' # what does a 'subject' look like ?
# SPECIAL_CHARS = r'[^a-zA-Z0-9\s]' # what does a 'special character' look like ?
SENDER = '(?<=from:\\s).*' # what does a 'sender' look like ?
SENDER_DOMAIN = '(?<=@)[\\w|.]+' # identify domain names in 'sender'


class EmailClassifier:

	def train(self, root_dir, datafiles_list):

		# assuming the ground truth labels are available
		labels_file = os.path.join(root_dir, "labels.csv")
		if not os.path.isfile(labels_file):
			raise FileNotFoundError("labels.csv cannot be found in {}".format(root_dir))
		labels = load_labels(labels_file)

		# declare a counter for every feature
		spam_words = Counter()
		spam_subjects = Counter()
		spam_domains = Counter()
		ham_words = Counter()
		ham_subjects = Counter()
		ham_domains = Counter()

		# process data files specified by datafiles_list
		for df in tqdm(datafiles_list, desc="training model"):

			# stupid way to get the right path
			filename = os.path.join("/".join(root_dir.split("/")[:-1]), df)
			label = labels.get(df, None)

			# extract features of current file
			with open(filename, 'r', encoding='ISO-8859-1') as f:
				mail = f.read().lower()

				words = re.findall(SIGNIFICANT_WORDS, mail)
				subject = re.search(SUBJECT_WORDS, mail)
				subject = subject.group(0).split() if subject is not None else ["NONE"]
				sender = re.search(SENDER, mail)
				sender = sender.group(0) if sender is not None else ""
				sender_domain = re.search(SENDER_DOMAIN, sender)
				sender_domain = sender_domain.group(0).split(".") if sender_domain is not None else ["NONE"]

				# build class features
				if label == "spam":
					spam_words += Counter(words)
					spam_subjects += Counter(subject)
					spam_domains += Counter(sender_domain)

					spam_words = Counter(dict(spam_words.most_common(WORDLIST_LENGTH)))
					spam_subjects = Counter(dict(spam_subjects.most_common(WORDLIST_LENGTH)))
					spam_domains = Counter(dict(spam_domains.most_common(WORDLIST_LENGTH)))

				elif label == "ham":
					ham_words += Counter(words)
					ham_subjects += Counter(subject)
					ham_domains += Counter(sender_domain)

					ham_words = Counter(dict(ham_words.most_common(WORDLIST_LENGTH)))
					ham_subjects = Counter(dict(ham_subjects.most_common(WORDLIST_LENGTH)))
					ham_domains = Counter(dict(ham_domains.most_common(WORDLIST_LENGTH)))

				if LOG_LEVEL == 1:
					print("{} - {}".format(df, label))

		self.spam_words = spam_words
		self.spam_subjects = spam_subjects
		self.spam_domains = spam_domains


		self.ham_words = ham_words
		self.ham_subjects = ham_subjects
		self.ham_domains = ham_domains

		print("Training finished ({} training instances)".format(len(datafiles_list)))

	def predict(self, filename):
		"""Make a prediction for a given document."""

		ham_score, spam_score = 0, 0

		# extract features of current file
		with open(filename, 'r', encoding='ISO-8859-1') as f:
			mail = f.read().lower()

			words = Counter(re.findall(SIGNIFICANT_WORDS, mail))
			subject = re.search(SUBJECT_WORDS, mail)
			subject_words = Counter(subject.group(0).split() if subject is not None else ["NONE"])
			sender = re.search(SENDER, mail)
			sender = sender.group(0) if sender is not None else ""
			sender_domain = re.search(SENDER_DOMAIN, sender)
			sender_domain = Counter(sender_domain.group(0).split(".") if sender_domain is not None else ["NONE"])

			# compute class similarity by intersection with model features
			spam_words_similarity = words & self.spam_words
			spam_subject_similarity = subject_words & self.spam_subjects
			spam_domain_similarity = sender_domain & self.spam_domains

			ham_words_similarity = words & self.ham_words
			ham_subject_similarity = subject_words & self.ham_subjects
			ham_domain_similarity = sender_domain & self.ham_domains

			# add weights to features
			max_domain_score = max([max(spam_domain_similarity.values(), default=0), max(ham_domain_similarity.values(), default=0)])
			max_subject_score = max([max(spam_subject_similarity.values(), default=0), max(ham_subject_similarity.values(), default=0)])
			max_words_score = max([max(spam_words_similarity.values()), max(ham_words_similarity.values())])
			max_score = max(max_domain_score, max_subject_score, max_words_score)

			alpha_domain = max_score / max_domain_score if max_domain_score is not 0 else 0
			alpha_subject = max_score / max_subject_score if max_subject_score is not 0 else 0
			alpha_words = max_score / max_words_score if max_words_score is not 0 else 0

			# assign score by linear combination of class similarities
			spam_score = alpha_words*sum(spam_words_similarity.values()) + alpha_domain*sum(spam_domain_similarity.values()) #+ alpha_subject*sum(spam_subject_similarity.values()) 
			ham_score = alpha_words*sum(ham_words_similarity.values()) + alpha_domain*sum(ham_domain_similarity.values()) #+ alpha_subject*sum(ham_subject_similarity.values())

		# make decision
		return "spam" if spam_score > ham_score else "ham"


	def predict_set(self, root_dir, datafiles_list, output_file):
		"""Makes predictions for a set of documents in a given dir."""

		print("Making predictions for {} and writing them to {}".format(root_dir, output_file))

		with open(output_file, "w") as f:
			f.write("Id,Label\n")
			for df in tqdm(datafiles_list, desc="making predictions"):
				filename = os.path.join("/".join(root_dir.split("/")[:-1]), df)
				label = self.predict(filename)
				f.write("{},{}\n".format(df, label))
				if LOG_LEVEL > 0:
					print("{} - {}".format(df, label))


	def save(self, output_file):
		with open(output_file, 'wb') as f:	
			pickle.dump(self.__dict__, f)

		print("Model saved to {}".format(output_file))


	def load(self, model_path):
		print("Loading model from {}".format(model_path))
		with open(model_path, 'rb') as f:
			self.__dict__.update(pickle.load(f))


	def __str__(self):
		model = "\n Model \n" + "-------" + "\n"
		model += "spam_words : " + str(self.spam_words.most_common(10)) + "\n\n"
		model += "ham_words : " + str(self.ham_words.most_common(10)) + "\n\n"
		model += "spam_subjects : " + str(self.spam_subjects.most_common(10)) + "\n\n"
		model += "ham_subjects : " + str(self.ham_subjects.most_common(10)) + "\n\n"
		return model


def argument_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--mode",
		default="predict",
		choices=["predict", "eval", "train"],
		help="Run script in a specific mode.",
	)

	parser.add_argument(
		"--strategy",
		default="none",
		choices=["none", "split", "cross"],
		help="How to split the labeled set during training."
	)

	parser.add_argument("--data", help="Directory path to the data.")
	parser.add_argument("--model", help="File path from which to load a trained model.")
	parser.add_argument("--output", help="File path to save the file produced by this script.")
	parser.add_argument("--predictions", help="File path to the saved predictions for evaluation.")
	parser.add_argument("--ground_truth", help="File path to the ground truth file.")

	return parser


def load_labels(labels_file):
	"""Load Id and Label pairs from a csv file and return them as a dictionary."""
	labels = dict()
	with open(labels_file, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)  # skip header line
		for row in reader:
			labels[row[0]] = row[1]
	return labels


def load_datafile_list(root_dir, mode):
	"""Load a list of data file paths in {root_dir}/xxx/yyy format."""
	datafiles_list = []
	subdirs_list = [x[0] for x in os.walk(root_dir)]
	if len(subdirs_list) > 0:
		for subdir in sorted(subdirs_list)[1:]:
			# Keep the upper two folder names as prefix to the filename
			prefix = "/".join(subdir.split("/")[-2:])
			for _, _, files in os.walk(subdir):
				for name in sorted(files):
					datafiles_list.append("/".join([prefix, name]))

	return datafiles_list


def digitize_label(label):
	if label == 'spam':
		return 1
	elif label == 'ham':
		return 0
	return None


def evaluate_predictions(ground_path, preds_path):
	"""Evaluate predictions against a ground truth file."""

	# Load predictions from file into a dictionary
	predictions = dict()
	with open(preds_path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)  # skip header line
		for row in reader:
			id, label = row[0], digitize_label(row[1])
			predictions[id] = label

	# Evaluate predictions against the ground truth
	y_true = []
	y_pred = []
	false_positives = []
	false_negatives = []

	with open(ground_path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)  # skip header line
		for row in reader:
			id, label = row[0], digitize_label(row[1])
			if id not in predictions:
				if LOG_LEVEL > 1:
					# raise KeyError("Missing prediction for document {}".format(id))
					print("No prediction for document {}".format(id))
			else:
				y_true.append(label)
				y_pred.append(predictions[id])
				
				# Identify error sources
				if label != predictions[id]:
					if predictions[id]:
						false_positives.append('df=="'+id+'"')
					else:
						false_negatives.append('df=="'+id+'"')

	print("Evaluation results:")
	print("-------------------")
	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
	print("Accuracy:            {:05.3f}".format(
		(tp + tn) / (tp + tn + fp + fn)))
	print("Precision:           {:05.3f}".format(tp / (tp + fp)))
	print("False positive rate: {:05.3f}".format(fp / (fp + tn)))

	return false_positives, false_negatives


def merge_predictions(file_list, output_file):
	with open(output_file, 'w') as output:
		output.write("Id,Label\n")
		for preds_file in file_list:
			with open(preds_file) as preds:
				next(preds)
				for line in preds:
					output.write(line)


if __name__ == "__main__":
	parser = argument_parser()
	args = parser.parse_args()
	datafiles_list = None
	data_path = None
	model_path = None
	preds_path = None
	ground_path = None
	strategy = None

	# Check if dataset, model, predictions, or ground truth file are present:
	mode = args.__dict__["mode"]
	strategy = args.__dict__["strategy"]

	if args.__dict__["data"] is not None:
		data_path = str(args.__dict__["data"])
		datafiles_list = load_datafile_list(data_path, mode)

	if mode == "train":
		model_path = str(args.__dict__["output"])
		ground_path = os.path.join(str(args.__dict__["data"]), "labels.csv")
	elif mode == "predict":
		model_path = str(args.__dict__["model"])
		preds_path = str(args.__dict__["output"])
		preds_path = str(args.__dict__["output"])
	elif mode == "eval":
		preds_path = str(args.__dict__["predictions"])
		ground_path = str(args.__dict__["ground_truth"])

	# Check if provided paths/files exist
	if model_path and mode == "predict":
		if not os.path.isfile(model_path):
			raise argparse.ArgumentTypeError(
				"Invalid path: {}".format(model_path))
	if preds_path and mode == "eval":
		if not os.path.isfile(preds_path):
			raise argparse.ArgumentTypeError(
				"Invalid path: {}".format(preds_path))
	if ground_path:
		if not os.path.isfile(ground_path):
			raise argparse.ArgumentTypeError(
				"Invalid path: {}".format(ground_path))

	if mode == "train" and model_path:
		model = EmailClassifier()

		if strategy == "split":
			split_length = int(SPLIT_RATIO*len(datafiles_list))
			datafiles_list = datafiles_list[-split_length:]
			model.train(data_path, datafiles_list)
			model.save(model_path)
		
		elif strategy == "cross":
			print("Generating folds, might take a while ...")
			for fold in range(KFOLDS):
				split_size = int(len(datafiles_list) / KFOLDS)
				test_split = datafiles_list[fold*split_size:(fold+1)*split_size]
				fold_datafiles_list = [filename for filename in datafiles_list if filename not in test_split]
				fold_model_path = model_path + "_fold{}".format(fold)
				model.train(data_path, fold_datafiles_list)
				model.save(fold_model_path)

		elif strategy == "none":
			model.train(data_path, datafiles_list)
			model.save(model_path)

	elif mode == "predict" and preds_path:
		if strategy == "split":
			model = EmailClassifier()
			model.load(model_path)
			split_length = int((1-SPLIT_RATIO)*len(datafiles_list))
			datafiles_list = datafiles_list[:split_length]
			model.predict_set(data_path, datafiles_list, preds_path)

		elif strategy == "cross":
			preds_files = []
			for fold in range(KFOLDS):
				split_size = int(len(datafiles_list) / KFOLDS)
				test_split = datafiles_list[fold*split_size:(fold+1)*split_size]

				fold_preds_path = preds_path + "_fold{}".format(fold)
				preds_files.append(fold_preds_path)

				fold_model_path = model_path + "_fold{}".format(fold)
				model = EmailClassifier()
				model.load(fold_model_path)
				model.predict_set(data_path, test_split, fold_preds_path)

			merge_predictions(preds_files, preds_path)

		elif strategy == "none":
			model.predict_set(data_path, datafiles_list, preds_path)

	elif mode == "eval":
		false_positives, false_negatives = evaluate_predictions(ground_path, preds_path)
		
		## Identify error sources
		with open('false_positives.txt', 'w') as f:
			f.writelines("%s\n" % item  for item in false_positives)

		with open('false_negatives.txt', 'w') as f:
			f.writelines("%s\n" % item  for item in false_negatives)

	else:
		print("Request not understood (invalid mode). Exiting.")
		sys.exit()

