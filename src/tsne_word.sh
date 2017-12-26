[ -f path.sh ] && . ./path.sh
path=/home/nanhao/Audio2vec

train_file=$path/$1
word_dic=$path/words.txt
target_words=$path/test_words

$path/src/tsne_eval.py $train_file $word_dic $target_words
