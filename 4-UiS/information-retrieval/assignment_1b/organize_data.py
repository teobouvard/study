import argparse
import csv
import os
import shutil
from tqdm import tqdm
from sklearn.datasets import load_files
import pickle

def organize_directories(input_dir, output_dir):
    """ Create a directory structure required by sklearn load_files. """
    labels = os.path.join(input_dir, "train", "labels.csv")

    with open(labels, 'r') as f:
        file_length = sum(1 for row in f)
        f.seek(0)

        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip header line
        for row in tqdm(reader, desc="Reading labels file", total=file_length):
            id, label = row[0], row[1]
            if not os.path.isdir(os.path.join(output_dir, label)):
                os.mkdir(os.path.join(output_dir, label))
            else:
                shutil.copyfile(os.path.join(input_dir, id), os.path.join(output_dir, label, str(id).replace("/",".")))


def pickle_data(input_dir, output_dir):
    """ Pickles the data to access it more quickly from the classifier. """
    train_pickle_path = os.path.join(output_dir, "train_data.pkl")
    test_pickle_path = os.path.join(output_dir, "test_data.pkl")

    print("Pickling formatted data to {}".format(output_dir))

    with open(train_pickle_path, 'wb') as f:
        pickle.dump(load_files(output_dir, encoding='ISO-8859-1'), f)
    
    with open(test_pickle_path, 'wb') as f:
        pickle.dump(load_files(os.path.join(input_dir, "test"), encoding='ISO-8859-1'), f)


def argument_parser():
    parser = argparse.ArgumentParser(description="Organize data so it can be directly loaded by sklearn")
    parser.add_argument("--input", help="Data input directory")
    parser.add_argument("--output", help="Data output directory")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()

    input_dir = str(args.input)
    output_dir = str(args.output)

    if not os.path.isdir(input_dir):
        raise argparse.ArgumentTypeError("Input directory must be a valid path")
    elif not os.path.isdir(output_dir):
        raise argparse.ArgumentTypeError("Output directory must be a valid path")

    organize_directories(input_dir, output_dir)
    pickle_data(input_dir, output_dir)

