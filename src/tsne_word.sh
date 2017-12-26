[ -f path.sh ] && . ./path.sh
src_path=/home/nanhao/Audio2vec
path=/nfs/Mazu/nanhao/yeeee

#train_file=$path/$1
word_dic=$path/words.txt
target_words=$path/test_words

#$path/src/tsne_eval.py $train_file $word_dic $target_words
python3 $src_path/src/tsne_eval.py $path/test_AE_words_words $word_dic $target_words
python3 $src_path/src/tsne_eval.py $path/test_AE_spks_words $word_dic $target_words
