#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 4 ] ; then 
  echo "usage: train.sh <p_hidden_dim> <s_hidden_dim> <CUDA_DEVICE> <model type>"
  echo "model_type: default, noGANspk"
  echo "e.g. train.sh 128 128 0 default"
  exit 1
fi

init_lr=0.0005
batch_size=32
seq_len=50
feat_dim=39
stack_num=3
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee
p_dim=$1
s_dim=$2
device_id=$3
n_epochs=$4
model_type=$5

if [$model_type != default] && [$model_type != noGANspk] ; then
  echo "Invalid model_type!"
  exit 1
fi

model_dir=$path/exp/model_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
log_dir=$path/exp/log_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir
### testing ###
export CUDA_VISIBLE_DEVICES=$device_id
python $path/src/audio2vec_eval.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --stack_num=$stack_num $tf_log_dir $tf_model_dir $feat_dir/test_AE.scp 
$feat_dir $model_type $feat_dir/words_AE_test $feat_dir/spks_AE_test 2> $tf_log_dir/test.log 

$path/src/trans_dir_to_file.py $feat_dir/words_AE_test $feat_dir/test_AE_words
$path/src/trans_dir_to_file.py $feat_dir/spks_AE_test $feat_dir/test_AE_spks
