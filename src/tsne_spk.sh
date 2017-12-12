[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Desktop/Audio2Vec

train_file=$path/$1
spk_dic=$path/spks.txt
target_spks=$path/test_spks

$path/src/tsne_eval.py $train_file $spk_dic $target_spks
