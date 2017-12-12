[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Desktop/Audio2Vec

train_file=$path/test_AE_spks
spk_dic=$path/spks.txt
target_spks=$path/test_spks

$path/src/PCA_eval_spk.py $train_file $spk_dic $target_spks --pca_dim=2
