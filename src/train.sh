#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 5 ] ; then 
  echo "usage: train.sh <p_hidden_dim> <s_hidden_dim> <CUDA_DEVICE> <n_epochs for training> <model type>"
  echo "model_type: default, noGANspk"
  echo "e.g. train.sh 128 128 0 20 default"
  exit 1
fi

init_lr=0.0001
batch_size=32
seq_len=50
feat_dim=39
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee
p_dim=$1
s_dim=$2
device_id=$3
n_epochs=$4
model_type=$5
model_dir=$path/exp/model_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
log_dir=$path/exp/log_lr${init_lr}_negspk0.1_$p_dim\_$s_dim\_$model_type
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir

### training ###
export CUDA_VISIBLE_DEVICES=$device_id
python $path/src/audio2vec_train.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --n_epochs=$n_epochs $tf_log_dir $tf_model_dir $feat_dir/train_AE.scp \
  $feat_dir/test_AE.scp $feat_dir $model_type 2> $tf_log_dir/train.log
