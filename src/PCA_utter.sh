[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Desktop/Audio2Vec

train_file=$path/test_AE_utters_400
utter_dic=$path/utters.txt
target_utters=$path/test_utters

$path/src/PCA_eval_utter.py $train_file $utter_dic $target_utters
