#!/usr/bin/env bash

python3 model.py --dataset dataset/$1/dataset.pkl --num-epochs 60 \
--word-embeddings data/embedding/character.vec \
--log-dir result/$1 \
--dropout 0.2 \
--learning-rate 0.01 \
--learning-rate-decay 0.9 \
--hidden-dim 100 \
--dynet-seed $RANDOM \
--bigram \
--skip-dev \
${@:2}