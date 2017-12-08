#!/bin/bash

[ -f path.sh ] && . ./path.sh

# if [ $# != 6 ] ; then 
#   echo "usage: train_origin.sh <feat_dir> <hidden_dim> <model_dir> <log_dir> <CUDA_DEVICE> <max step for training>"
#   echo "e.g. train_origin.sh ./feat 100 ./model ./logs 1 80000"
#   echo "The feat dir should conatin:"
#   echo "cmvned_feats.ark all.scp all_prons words.txt"
#   exit 1
# fi
init_lr=0.0005
batch_size=32
seq_len=50
feat_dim=39
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee
p_dim=$1
s_dim=$2
model_dir=$path/exp/model_lr${init_lr}_negspk0.1_$p_dim\_$s_dim
log_dir=$path/exp/log_lr${init_lr}_negspk0.1_$p_dim\_$s_dim
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log
device_id=$3
n_epochs=$4

mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir

### training ###
export CUDA_VISIBLE_DEVICES=$device_id
python $path/src/audio2vec_train.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --n_epochs=$n_epochs $tf_log_dir $tf_model_dir $feat_dir/train_AE.scp \
  $feat_dir/test_AE.scp $feat_dir 2> $tf_log_dir/train.log
