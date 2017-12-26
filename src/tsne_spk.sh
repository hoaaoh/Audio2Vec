[ -f path.sh ] && . ./path.sh
src_path=/home/nanhao/Audio2vec
path=/nfs/Mazu/nanhao/yeeee

#train_file=$path/$1
spk_dic=$path/spks.txt
target_spks=$path/test_spks

#$path/src/tsne_eval.py $train_file $spk_dic $target_spks
python3 $src_path/src/tsne_eval.py $path/test_AE_words_spks $spk_dic $target_spks
python3 $src_path/src/tsne_eval.py $path/test_AE_spks_spks $spk_dic $target_spks
