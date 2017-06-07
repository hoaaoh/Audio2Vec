#!/bin/bash -ex

./src/tsne_eval.py --ave_num=10 --sample_num=30 --tsne_dim=2 \
		../baseline/words/train_list ../baseline/words/words.txt \
		./test_words



