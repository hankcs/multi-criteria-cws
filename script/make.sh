#!/usr/bin/env bash

python3 ./make_dataset.py --training-data  data/$1/bmes/train-all.txt --dev-data data/$1/bmes/dev.txt \
--test-data data/$1/bmes/test.txt -o dataset/$1/dataset.pkl