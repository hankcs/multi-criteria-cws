# -*- coding:utf-8 -*-
# Filename: statistics.py
# Author：hankcs
# Date: 2017-08-18 下午9:24

import os
import sys
import tempfile
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", required=True, dest="dataset", help="Dataset name (eg. pku)")
# parser.add_argument("--output", required=True, dest="test_out", help="Test output .txt file")
# parser.add_argument("--joint", dest="joint", action="store_true", help="Is joint learning outputs")
# options = parser.parse_args()
# 'data/{}/raw/train-all.txt'.format(dataset)
from collections import Counter


def count(file):
    size_sentences = 0
    size_words = 0
    size_chars = 0
    dict_word = Counter()
    dict_char = Counter()
    with open(file) as src:
        for line in src:
            size_sentences += 1
            sentence = line.split()
            for word in sentence:
                size_words += 1
                dict_word[word] += 1
                for char in word:
                    size_chars += 1
                    dict_char[char] += 1

    return size_sentences, size_words, size_chars, dict_word, dict_char


def analysis(dataset):
    train_size_sentences, train_size_words, train_size_chars, train_dict_word, train_dict_char = count(
        'data/{}/raw/train-all.txt'.format(dataset))
    test_size_sentences, test_size_words, test_size_chars, test_dict_word, test_dict_char = count(
        'data/{}/raw/test.txt'.format(dataset))

    freq = 0
    for oov in (set(
            test_dict_word.keys()) - set(
        train_dict_word.keys())):
        freq += test_dict_word[oov]

    print('{:^8}\t{:^8}\t{:^8}\t{:^8}\t{:^8}\t{:^8}\t{:^8}'.format(dataset, 'Sents', 'Words', 'Chars',
                                                                   'Word Types',
                                                                   'Char Types', 'OOV Rate'))
    print('{:^8}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t-'.format('Train',
                                                                              train_size_sentences / 1000,
                                                                              train_size_words / 1000,
                                                                              train_size_chars / 1000,
                                                                              len(train_dict_word) / 1000,
                                                                              len(train_dict_char) / 1000))
    print('{:^8}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t{:^8.1f}\t{:^.2f}%'.format('Test',
                                                                                     test_size_sentences / 1000,
                                                                                     test_size_words / 1000,
                                                                                     test_size_chars / 1000,
                                                                                     len(test_dict_word) / 1000,
                                                                                     len(test_dict_char) / 1000,
                                                                                     freq / sum(
                                                                                         test_dict_word.values()) * 100))


for dataset in 'pku', 'msr', 'as', 'cityu':
    analysis(dataset)
    print()
