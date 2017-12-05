import os
import sys

"""
Reads in tab separated files to make the dataset
Output a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2i: Dict mapping tags to indices
c2i: Dict mapping characters to indices
"""

import codecs
import argparse
import pickle
import collections
from utils import get_processing_word, read_pretrained_embeddings, is_dataset_tag, make_sure_path_exists

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_CHAR = "<*>"


def read_file(filename, w2i, t2i, c2i, max_iter=sys.maxsize, processing_word=get_processing_word(lowercase=False)):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars 
    as it sees them.
    """
    instances = []
    vocab_counter = collections.Counter()
    niter = 0
    with codecs.open(filename, "r", "utf-8") as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    niter += 1
                    if max_iter is not None and niter > max_iter:
                        break
                    instances.append(Instance(words, tags))
                    words, tags = [], []
            else:
                word, tag = line.split()
                word = processing_word(word)
                vocab_counter[word] += 1
                if word not in w2i:
                    w2i[word] = len(w2i)
                if tag not in t2i:
                    t2i[tag] = len(t2i)
                if is_dataset_tag(word):
                    if word not in c2i:
                        c2i[word] = len(c2i)
                else:
                    for c in word:
                        if c not in c2i:
                            c2i[c] = len(c2i)
                words.append(w2i[word])
                tags.append(t2i[tag])
    return instances, vocab_counter


parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()

w2i = {}  # mapping from word to index
t2i = {}  # mapping from tag to index
c2i = {}
output = {}
print('Making training dataset')
output["training_instances"], output["training_vocab"] = read_file(options.training_data, w2i, t2i, c2i)
print('Making dev dataset')
output["dev_instances"], output["dev_vocab"] = read_file(options.dev_data, w2i, t2i, c2i)
print('Making test dataset')
output["test_instances"], output["test_vocab"] = read_file(options.test_data, w2i, t2i, c2i)

# Add special tokens / tags / chars to dicts
w2i[UNK_TAG] = len(w2i)
t2i[START_TAG] = len(t2i)
t2i[END_TAG] = len(t2i)
c2i[UNK_TAG] = len(c2i)

output["w2i"] = w2i
output["t2i"] = t2i
output["c2i"] = c2i

# Read embedding
if options.word_embeddings:
    output["word_embeddings"] = read_pretrained_embeddings(options.word_embeddings, w2i)

make_sure_path_exists(os.path.dirname(options.output))

print('Saving dataset to {}'.format(options.output))
with open(options.output, "wb") as outfile:
    pickle.dump(output, outfile)

with codecs.open(os.path.dirname(options.output) + "/words.txt", "w", "utf-8") as vocabfile:
    for word in w2i.keys():
        vocabfile.write(word + "\n")

with codecs.open(os.path.dirname(options.output) + "/chars.txt", "w", "utf-8") as vocabfile:
    for char in c2i.keys():
        vocabfile.write(char + "\n")
