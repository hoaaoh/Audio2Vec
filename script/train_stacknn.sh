#!/bin/bash -ex

[ -f path.sh ] && . path.sh || exit 1

if [ $# != 3 ];
then
    echo "usage: ./script/train_stckrnn.sh <STACK NUM> <HIDDEN DIM> <BATCHSIZE> <GPU_NUM>"
    exit 1
fi

dim=100
log_dir=/home_local/hoa/exp/log
model_dir=/home_local/hoa/exp/model
feat_dir=/home_local/hoa/libri_feat
stack_num=$1
hidden_dim=$2
batch_size=$3
export CUDA_VISIBLE_DEVICES=$4


mkdir -p $log_dir/$dim $model_dir/$dim
mkdir -p $log_dir/$dim/"$stack_num"_"$batch_size" $model_dir/$dim/"$stack_num"_"$batch_size"
log_dir=$log_dir/$dim/"$stack_num"_"$batch_size"
feat_dir=$model_dir/$dim/"$stack_num"_"$batch_size"
audio2vec_train.py --init_lr=1 --decay_rate=1000 --hidden_dim=$hidden_dim --stack_num=$stack_num --batch_size=$batch_size --max_step=100000  \
	$log_dir $model_dir $feat_dir/train_AE.scp

