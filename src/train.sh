#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 6 ] ; then 
  echo "usage: train.sh <lr> <p_hidden_dim> <s_hidden_dim> <CUDA_DEVICE> <model type> <n_epochs for training>"
  echo "model_type: default, noGAN, noGANspk"
  echo "e.g. train.sh 0.0005 128 128 0 default 20"
  exit 1
fi

batch_size=64
seq_len=70
feat_dim=39
stack_num=3
path=/home/grtzsohalf/Audio2Vec
feat_dir=/nfs/YueLao/grtzsohalf/yeeee/English
init_lr=$1
p_dim=$2
s_dim=$3
device_id=$4
model_type=$5
n_epochs=$6

if [ "$model_type" != "default" ] && [ "$model_type" != "noGAN" ] && [ "$model_type" != "noGANspk" ] ; then
  echo "Invalid model_type!"
  exit 1
fi

exp_dir=/home_local/grtzsohalf/yeeee/interpolate_exp
mkdir -p $exp_dir
model_dir=$exp_dir/model_lr${init_lr}_$p_dim\_$s_dim\_$model_type
log_dir=$exp_dir/log_lr${init_lr}_$p_dim\_$s_dim\_$model_type

#model_dir=$path/exp/model_lr${init_lr}_$p_dim\_$s_dim\_$model_type
#log_dir=$path/exp/log_lr${init_lr}_$p_dim\_$s_dim\_$model_type
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir

### training ###
export CUDA_VISIBLE_DEVICES=$device_id
python3 $path/src/audio2vec_train.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$p_dim --s_hidden_dim=$s_dim --n_epochs=$n_epochs --stack_num=$stack_num $tf_log_dir $tf_model_dir \
  $feat_dir/train_AE.scp $feat_dir/test_AE.scp $feat_dir $model_type 2> $tf_log_dir/train.log
