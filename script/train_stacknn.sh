#!/bin/bash -ex

[ -f path.sh ] && . path.sh || exit 1

if [ $# != 3 ];
then
    echo "usage: ./script/train_stckrnn.sh <STACK_NUM> <GPU_NUM>"
    exit 1
fi

dim=100
log_dir=/home_local/hoa/exp/log
model_dir=/home_local/hoa/exp/model
feat_dir=/home_local/hoa/libri_feat
stack_num=$2

export CUDA_VISIBLE_DEVICES=$3
mkdir -p $log_dir/$dim $model_dir/$dim

audio2vec_train.py --init_lr=1 --decay_rate=1000 --hidden_dim=100 --stack_num=$stack_num --batch_size=500 --max_step=100000  \
	$log_dir/$dim $model_dir/$dim $feat_dir/train_AE.scp

