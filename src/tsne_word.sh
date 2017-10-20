[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee

train_file=$feat_dir/test_AE_words_400
word_dic=$feat_dir/words.txt
target_words=$path/test_words

$path/src/tsne_eval.py $train_file $word_dic $target_words
