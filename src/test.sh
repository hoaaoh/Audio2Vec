#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 5 ] ; then 
  echo "usage: train.sh <lr> <p_hidden_dim> <s_hidden_dim> <CUDA_DEVICE> <model type>"
  echo "model_type: default, noGAN, noGANspk"
  echo "e.g. train.sh 0.0005 128 128 0 default"
  exit 1
fi

batch_size=64
seq_len=70
feat_dim=39
stack_num=3
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee
init_lr=$1
p_dim=$2
s_dim=$3
device_id=$4
model_type=$5

if [ "$model_type" != "default" ] && [ "$model_type" != "noGAN" ] && [ "$model_type" != "noGANspk" ] ; then
  echo "Invalid model_type!"
  exit 1
fi

exp_dir=/home_local/grtzsohalf/yeeee/exp
mkdir -p $exp_dir
model_dir=$exp_dir/model_lr${init_lr}_$p_dim\_$s_dim\_$model_type
log_dir=$exp_dir/log_lr${init_lr}_$p_dim\_$s_dim\_$model_type

#model_dir=$path/exp/model_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
#log_dir=$path/exp/log_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir

mkdir -p $feat_dir/words_words_AE_test
mkdir -p $feat_dir/words_spks_AE_test
mkdir -p $feat_dir/spks_words_AE_test
mkdir -p $feat_dir/spks_spks_AE_test

mkdir -p $feat_dir/words_words_AE_train
mkdir -p $feat_dir/words_spks_AE_train
mkdir -p $feat_dir/spks_words_AE_train
mkdir -p $feat_dir/spks_spks_AE_train

### testing ###
export CUDA_VISIBLE_DEVICES=$device_id
python3 $path/src/audio2vec_eval.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --stack_num=$stack_num $tf_log_dir $tf_model_dir $feat_dir/test_AE.scp \
$feat_dir $model_type $feat_dir/words_words_AE_test $feat_dir/words_spks_AE_test \
$feat_dir/spks_words_AE_test $feat_dir/spks_spks_AE_test $feat_dir/phonetic_test

python3 $path/src/audio2vec_eval.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --stack_num=$stack_num $tf_log_dir $tf_model_dir $feat_dir/train_AE.scp \
$feat_dir $model_type $feat_dir/words_words_AE_train $feat_dir/words_spks_AE_train \
$feat_dir/spks_words_AE_train $feat_dir/spks_spks_AE_train $feat_dir/phonetic_train 

python3 $path/src/trans_dir_to_file.py $feat_dir/words_words_AE_train $feat_dir/train_AE_words_words
#python3 $path/src/trans_dir_to_file.py $feat_dir/words_spks_AE_train $feat_dir/train_AE_words_spks
#python3 $path/src/trans_dir_to_file.py $feat_dir/spks_words_AE_train $feat_dir/train_AE_spks_words
#python3 $path/src/trans_dir_to_file.py $feat_dir/spks_spks_AE_train $feat_dir/train_AE_spks_spks

python3 $path/src/trans_dir_to_file.py $feat_dir/words_words_AE_test $feat_dir/test_AE_words_words
#python3 $path/src/trans_dir_to_file.py $feat_dir/words_spks_AE_test $feat_dir/test_AE_words_spks
#python3 $path/src/trans_dir_to_file.py $feat_dir/spks_words_AE_test $feat_dir/test_AE_spks_words
#python3 $path/src/trans_dir_to_file.py $feat_dir/spks_spks_AE_test $feat_dir/test_AE_spks_spks
