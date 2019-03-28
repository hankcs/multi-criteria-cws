import argparse
import pickle
import collections
import logging
import math
import os
import random
from sys import maxsize

import dynet as dy 
import numpy as np
import time

import sys

import utils

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"

DEFAULT_WORD_EMBEDDING_SIZE = 100
# TRAIN_LIMIT = 1000000000
# TRAIN_LIMIT = 10000000
DEBUG_SCALE = 200


class BiLSTM_CRF:
    def __init__(self, tagset_size, num_lstm_layers, hidden_dim, word_embeddings, no_we_update, use_char_rnn,
                 char_embeddings, char_hidden_dim, margins, lowercase_words, vocab_size=None,
                 word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE,
                 charset_size=None, char_embedding_dim=50, tie_two_embeddings=False, use_we=True):
        self.dropout = None
        self.model = dy.Model()
        self.tagset_size = tagset_size
        self.margins = margins
        self.we_update = not no_we_update
        self.lowercase_words = lowercase_words

        # Word embedding parameters
        self.use_we = use_we
        if use_we:
            if word_embeddings is not None:  # Use pretrained embeddings
                vocab_size = word_embeddings.shape[0]
                word_embedding_dim = word_embeddings.shape[1]
            self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
            if word_embeddings is not None:
                self.words_lookup.init_from_array(word_embeddings)
        else:
            self.words_lookup = None

        # bigram embeddings
        if options.bigram:
            self.bigram_lookup = self.model.add_lookup_parameters((len(b2i), word_embedding_dim))
            self.bigram_lookup.init_from_array(bigram_embeddings)

        # Char LSTM Parameters
        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            if char_embeddings is not None:
                charset_size = char_embeddings.shape[0]
                char_embedding_dim = char_embeddings.shape[1]
            self.char_embedding_dim = char_embedding_dim
            if tie_two_embeddings:
                self.char_lookup = self.words_lookup
            else:
                self.char_lookup = self.model.add_lookup_parameters((charset_size, self.char_embedding_dim))
                if char_embeddings is not None:
                    self.char_lookup.init_from_array(char_embeddings)
            self.char_bi_lstm = dy.BiRNNBuilder(1, self.char_embedding_dim, char_hidden_dim, self.model, dy.LSTMBuilder)

            # Cache char ids for each word for fast speed
            self.word_to_char_ids = dict()
            for word, word_id in w2i.items():
                # Note: use original casing ("word") for characters
                if utils.is_dataset_tag(word):
                    char_ids = [c2i[word]]
                else:
                    char_ids = [c2i[c] for c in word]
                self.word_to_char_ids[word_id] = char_ids

        # Word LSTM parameters
        if use_char_rnn:
            if use_we:
                input_dim = word_embedding_dim + char_hidden_dim
            else:
                input_dim = char_hidden_dim
        else:
            input_dim = word_embedding_dim
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)
        # Matrix that maps from Bi-LSTM output to num tags
        if options.bigram:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim + word_embedding_dim * 2))
        else:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(tagset_size)
        self.mlp_out = self.model.add_parameters((tagset_size, tagset_size))
        self.mlp_out_bias = self.model.add_parameters(tagset_size)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((tagset_size, tagset_size))

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)
        self.dropout = p

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()
        self.dropout = None

    def word_rep(self, word):
        '''
        :param word: index of word in lookup table
        '''

        if options.bigram:
            word = word[1]
            pass

        if self.use_char_rnn:
            # Note: use original casing ("word") for characters
            char_ids = self.word_to_char_ids[word]
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            if self.use_we:
                wemb = dy.lookup(self.words_lookup, word, update=self.we_update)
                return dy.concatenate([wemb, char_exprs[-1]])
            else:
                return char_exprs[-1]
        else:
            wemb = dy.lookup(self.words_lookup, word, update=self.we_update)
            return wemb

    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]

        lstm_out = self.bi_lstm.transduce(embeddings)

        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        if options.bigram:
            for rep, word in zip(lstm_out, sentence):
                bi1 = dy.lookup(self.bigram_lookup, word[0], update=self.we_update)
                bi2 = dy.lookup(self.bigram_lookup, word[1], update=self.we_update)
                if self.dropout is not None:
                    bi1 = dy.dropout(bi1, self.dropout)
                    bi2 = dy.dropout(bi2, self.dropout)
                score_t = O * dy.tanh(H * dy.concatenate(
                    [bi1,
                     rep,
                     bi2]) + Hb) + Ob
                scores.append(score_t)
        else:
            for rep in lstm_out:
                score_t = O * dy.tanh(H * rep + Hb) + Ob
                scores.append(score_t)

        return scores

    def score_sentence(self, observations, tags):
        if len(tags) == 0:
            tags = [t2i[NONE_TAG]] * len(observations)
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [t2i[START_TAG]] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[t2i[END_TAG]], tags[-1])
        return score

    def viterbi_loss(self, sentence, gold_tags, use_margins=True):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations, gold_tags, use_margins)
        if viterbi_tags != gold_tags:
            gold_score = self.score_sentence(observations, gold_tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def neg_log_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_dim(dy.transpose(dy.exp(scores - max_score_expr_broadcast)),[1]))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[t2i[START_TAG]] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[t2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations, gold_tags, use_margins):
        backpointers = []
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[t2i[START_TAG]] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tagset_size)]
        for gold, obs in zip(gold_tags, observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs

            # optional margin adaptation
            if use_margins and self.margins != 0:
                adjust = [self.margins] * self.tagset_size
                adjust[gold] = 0
                for_expr = for_expr + dy.inputVector(adjust)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[t2i[END_TAG]]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == t2i[START_TAG]
        # Return best path and best path's score
        return best_path, path_score

    def save(self, file_name):
        # members_to_save = []
        # members_to_save.append(self.words_lookup)
        # if (self.use_char_rnn):
        #     members_to_save.append(self.char_lookup)
        #     members_to_save.append(self.char_bi_lstm)
        # members_to_save.append(self.bi_lstm)
        # members_to_save.extend(utils.sortvals(self.lstm_to_tags_params))
        # members_to_save.extend(utils.sortvals(self.lstm_to_tags_bias))
        # members_to_save.extend(utils.sortvals(self.mlp_out))
        # members_to_save.extend(utils.sortvals(self.mlp_out_bias))
        # members_to_save.extend(utils.sortvals(self.transitions))
        self.model.save(file_name)

    def load(self, file_name):
        self.model.populate(file_name)


# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int,
                    help="Number of full passes through training set")
parser.add_argument("--batch-size", default=20, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--lstm-layers", default=1, dest="lstm_layers", type=int, help="Number of LSTM layers")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--learning-rate-decay", default=0.9, dest="learning_rate_decay", type=float,
                    help="Learning rate decay")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--no-we", dest="no_we", action="store_true", help="Word Embeddings aren't used")
parser.add_argument("--no-we-update", dest="no_we_update", action="store_true", help="Word Embeddings aren't updated")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN")
parser.add_argument("--char-embeddings", dest="char_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--char-embedding-dim", default=100, dest="char_embedding_dim", type=int,
                    help="Dimension of char embedding")
parser.add_argument("--char-hidden-dim", default=100, dest="char_hidden_dim", type=int,
                    help="Dimension of char LSTM hidden layer size")
parser.add_argument("--lowercase-words", dest="lowercase_words", action="store_true",
                    help="Words are all in lowercased form (characters stay the same)")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / serialized models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't serialize model")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always serialize model after every epoch")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set, would save some time")
parser.add_argument("--subset", dest="subset", help="Only train and test on a subset of the whole dataset")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--dynet-autobatch", help="Ignore this outside argument")
parser.add_argument("--dynet-seed", dest="dynet_seed", type=int, help="Ignore this outside argument")
parser.add_argument("--dynet-gpus", dest="dynet_gpus", type=int, help="Specify how many GPUs you want to use")
parser.add_argument("--dynet-weight-decay", dest="dynet_weight_decay", type=float,
                    help="If this value is set to wd, each parameter in the model is multiplied by (1-wd) after every "
                         "parameter update. This weight decay is similar to L2 regularization, but not exactly the "
                         "same.")
parser.add_argument("--clip-norm", dest="clip_norm", type=float, help="Gradient clipping")
parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--test", dest="test", action="store_true", help="Test mode")
parser.add_argument("--tie-two-embeddings", dest="tie_two_embeddings", action="store_true",
                    help="Tie word and char embeddings together")
parser.add_argument("--bigram", dest="bigram", action="store_true", help="Use bigram feature")
options = parser.parse_args()

task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
utils.make_sure_path_exists(root_dir)


def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger()

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

if options.debug:
    print("DEBUG MODE")
    options.num_epochs = 2

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
w2i = dataset["w2i"]
t2i = dataset["t2i"]
c2i = dataset["c2i"]
i2w = utils.to_id_list(w2i)  # Inverse mapping
i2t = utils.to_id_list(t2i)
i2c = utils.to_id_list(c2i)

training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]
test_instances = dataset["test_instances"]

if options.debug:
    training_instances = training_instances[0:DEBUG_SCALE]
    dev_instances = dev_instances[0:DEBUG_SCALE]
    test_instances = test_instances[0:DEBUG_SCALE]

if options.subset is not None:
    def pick_subset(dataset, subset):
        result = []
        tag = w2i['<' + subset + '>']
        for instance in dataset:
            if len(instance.sentence) > 2 and instance.sentence[0] == tag:
                result.append(instance)
        return result


    training_instances = pick_subset(training_instances, options.subset)
    dev_instances = pick_subset(dev_instances, options.subset)
    test_instances = pick_subset(test_instances, options.subset)

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
if options.bigram:
    def add_word(word):
        if word in w2i:
            pass
        id = len(w2i)
        w2i[word] = id
        i2w.append(word)
        assert len(w2i) == len(i2w)
        assert i2w[id] == word


    add_word(START_TAG)
    add_word(END_TAG)

if options.no_we:
    word_embeddings = None
elif options.word_embeddings is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
else:
    word_embeddings = dataset.get("word_embeddings")  # can be stored within dataset

if options.char_embeddings is not None:
    char_embeddings = utils.read_pretrained_embeddings(options.char_embeddings, c2i)
else:
    char_embeddings = dataset.get("char_embeddings")  # can be stored within dataset

# Tie embeddings up
if options.use_char_rnn and word_embeddings is not None and char_embeddings is not None and options.tie_two_embeddings:
    for c, i in c2i.items():
        if c not in w2i:
            wi = len(w2i)
            w2i[c] = wi
            word_embeddings = np.vstack([word_embeddings, char_embeddings[i]])
    c2i = w2i
    i2w = utils.to_id_list(w2i)  # Inverse mapping
    i2c = i2w

bigram_embeddings = None
if options.bigram:
    b2i = dict()


    def expand_instances(instances):

        def id_of_bigram(pre, cur):
            bi = (pre, cur)
            if bi not in b2i:
                b2i[bi] = len(b2i)

            return b2i[bi]

        for instance in instances:
            tuple_list = []
            sent = [w2i[START_TAG]] + instance.sentence + [w2i[END_TAG]]
            for index, word_id in enumerate(sent):
                if index == 0 or index == len(sent) - 1:
                    continue
                tuple_list.append(
                    (id_of_bigram(sent[index - 1], word_id), word_id, id_of_bigram(word_id, sent[index + 1])))
            del instance.sentence[:]
            instance.sentence.extend(tuple_list)


    expand_instances(training_instances)
    expand_instances(dev_instances)
    expand_instances(test_instances)
    bigram_embeddings = np.random.uniform(-0.8, 0.8, (len(b2i), word_embeddings.shape[1]))
    for (pre, cur), id in b2i.items():
        bigram_embeddings[id] = (word_embeddings[pre] + word_embeddings[cur]) / 2

tag_set_size = len(t2i)
model = BiLSTM_CRF(tag_set_size,
                   options.lstm_layers,
                   options.hidden_dim,
                   word_embeddings,
                   options.no_we_update,
                   options.use_char_rnn,
                   char_embeddings,
                   options.char_hidden_dim,
                   None,
                   options.lowercase_words,
                   vocab_size=len(w2i),
                   charset_size=len(c2i),
                   tie_two_embeddings=options.tie_two_embeddings
                   )

best_model_file_name = "{}/model.bin".format(root_dir)
# start training
if not options.test:
    if options.old_model:
        # incremental training
        print("Incremental training from old model: {}".format(options.old_model))
        model.load(options.old_model)
    trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
    if options.clip_norm:
        trainer.set_clip_threshold(options.clip_norm)
    logger.info("Training Algorithm: {}".format(type(trainer)))

    logger.info("Number training instances: {}".format(len(training_instances)))
    logger.info("Number dev instances: {}".format(len(dev_instances)))
    training_total_tokens = 0
    best_f1 = 0.
    for epoch in range(int(options.num_epochs)):
        logger.info("Epoch {} out of {}".format(epoch + 1, options.num_epochs))
        random.shuffle(training_instances)
        train_loss = 0.0
        train_total_instance = 0  # size of trained instances

        if options.dropout > 0:
            model.set_dropout(options.dropout)

        nbatches = (len(training_instances) + options.batch_size - 1) // options.batch_size

        bar = utils.Progbar(target=nbatches)
        for batch_id, batch in enumerate(utils.minibatches(training_instances, options.batch_size)):
            for idx, instance in enumerate(batch):
                if len(instance.sentence) == 0: continue
                train_total_instance += 1

                loss_expr = model.neg_log_loss(instance.sentence, instance.tags)
                # Forward pass
                loss = loss_expr.scalar_value()
                # Do backward pass
                loss_expr.backward()

                # Bail if loss is NaN
                if math.isnan(loss):
                    assert False, "NaN occured"

                train_loss += loss
                training_total_tokens += len(instance.sentence)

            trainer.update()
            if options.batch_size == 1 and batch_id % 10 != 0 and batch_id + 1 != train_total_instance:
                # online learning, don't print too often
                continue
            bar.update(batch_id + 1, exact=[("train loss", train_loss / train_total_instance)])

        # trainer.update_epoch(1)
        # trainer.learning_rate = options.learning_rate / (1 + (epoch + 1) * options.learning_rate_decay)
        trainer.learning_rate *= options.learning_rate_decay
        # print trainer.learning_rate

        train_loss = train_loss / len(training_instances)

        # Evaluate dev data
        if options.skip_dev:
            continue
        model.disable_dropout()
        dev_loss = 0.0
        dev_total_instance = 0
        dev_oov_total = 0
        total_wrong = 0
        total_wrong_oov = 0
        # PRF
        prf = utils.CWSEvaluator(t2i)
        prf_dataset = {}
        dev_batch_size = math.ceil(len(dev_instances) * 0.01)
        nbatches = (len(dev_instances) + dev_batch_size - 1) // dev_batch_size
        bar = utils.Progbar(target=nbatches)
        with open("{}/devout-epoch-{:02d}.txt".format(root_dir, epoch + 1), 'w', encoding='utf-8') as dev_writer:
            for batch_id, batch in enumerate(utils.minibatches(dev_instances, dev_batch_size)):
                for idx, instance in enumerate(batch):
                    sentence = instance.sentence
                    if len(sentence) == 0: continue

                    gold_tags = instance.tags
                    losses = model.neg_log_loss(sentence, gold_tags)
                    total_loss = losses.scalar_value()
                    _, out_tags = model.viterbi_loss(sentence, gold_tags, use_margins=False)

                    sentence = utils.restore_sentence(sentence)
                    dataset_name = None
                    if utils.is_dataset_tag(i2w[sentence[0]]):
                        dataset_name = i2w[sentence[0]][1:-1]
                        if dataset_name not in prf_dataset:
                            prf_dataset[dataset_name] = utils.CWSEvaluator(t2i)
                        sentence = sentence[1:-1]
                        gold_tags = gold_tags[1:-1]
                        out_tags = out_tags[1:-1]
                        prf_dataset[dataset_name].add_instance(gold_tags, out_tags)

                    prf.add_instance(gold_tags, out_tags)

                    gold_strings = utils.to_tag_strings(i2t, gold_tags)
                    obs_strings = utils.to_tag_strings(i2t, out_tags)

                    dev_total_instance += 1
                    dev_loss += total_loss
                    # output predict raw sentence
                    sent = [i2w[w][0] for w in sentence]
                    word_list = utils.bmes_to_words(sent, obs_strings)
                    raw_sentence_string = ' '.join(word_list)
                    if dataset_name is not None:
                        raw_sentence_string = '<{}> {} </{}>'.format(dataset_name, raw_sentence_string, dataset_name)
                    dev_writer.write(raw_sentence_string)
                    dev_writer.write('\n')

                bar.update(batch_id + 1, exact=[("dev loss", dev_loss / dev_total_instance)])

        dev_loss = dev_loss / len(dev_instances)

        # logging this epoch
        # print prf wrt dataset
        for dataset_name, performance in sorted(prf_dataset.items()):
            p = performance.result()
            logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format(dataset_name, p[0], p[1], p[2]))
        prf = prf.result()
        logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format('AVG', prf[0], prf[1], prf[2]))
        if prf[-1] > best_f1:
            best_f1 = prf[-1]
            logger.info("- new best score!")

            # Serialize model
            if not options.no_model:
                logger.info("Saving model to {}".format(best_model_file_name))
                model.save(best_model_file_name)
        elif options.always_model:
            logger.info("Saving model to {}".format(best_model_file_name))
            model.save(best_model_file_name)

# Evaluate test data (once)
logger.info("\n")
logger.info("Number test instances: {}".format(len(test_instances)))
if not options.skip_dev:
    if options.test:
        model.load(options.old_model)
    else:
        model.load(best_model_file_name)
model.disable_dropout()
test_correct = 0
test_total_instance = 0
test_oov_total = 0
prf = utils.CWSEvaluator(t2i)
prf_dataset = {}
test_batch_size = math.ceil(len(test_instances) * 0.01)
nbatches = (len(test_instances) + test_batch_size - 1) // test_batch_size
bar = utils.Progbar(target=nbatches)
with open("{}/testout.txt".format(root_dir), 'w', encoding='utf-8') as raw_writer:
    for batch_id, batch in enumerate(utils.minibatches(test_instances, test_batch_size)):
        for idx, instance in enumerate(batch):
            if len(instance.sentence) == 0: continue
            sentence = instance.sentence
            gold_tags = instance.tags
            _, out_tags = model.viterbi_loss(instance.sentence, gold_tags, use_margins=False)

            sentence = utils.restore_sentence(sentence)
            dataset_name = None
            if utils.is_dataset_tag(i2w[sentence[0]]):
                dataset_name = i2w[sentence[0]][1:-1]
                if dataset_name not in prf_dataset:
                    prf_dataset[dataset_name] = utils.CWSEvaluator(t2i)
                sentence = sentence[1:-1]
                gold_tags = gold_tags[1:-1]
                out_tags = out_tags[1:-1]
                prf_dataset[dataset_name].add_instance(gold_tags, out_tags)

            prf.add_instance(gold_tags, out_tags)
            gold_strings = utils.to_tag_strings(i2t, gold_tags)
            obs_strings = utils.to_tag_strings(i2t, out_tags)
            out_tags = out_tags

            test_total_instance += 1
            sent = [i2w[w][0] for w in sentence]
            # output predict raw sentence
            word_list = utils.bmes_to_words(sent, obs_strings)
            raw_sentence_string = ' '.join(word_list)
            if dataset_name is not None:
                raw_sentence_string = '<{}> {} </{}>'.format(dataset_name, raw_sentence_string, dataset_name)
            raw_writer.write(raw_sentence_string)
            raw_writer.write('\n')
        bar.update(batch_id + 1, exact=[("f1", prf.result()[-1])])

# print prf wrt dataset
for dataset_name, performance in sorted(prf_dataset.items()):
    p = performance.result()
    logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format(dataset_name, p[0], p[1], p[2]))
prf = prf.result()
logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format('AVG', prf[0], prf[1], prf[2]))
