[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Desktop/Audio2Vec

train_file=$path/$1
word_dic=$path/words.txt
target_words=$path/test_words

$path/src/PCA_eval.py $train_file $word_dic $target_words
