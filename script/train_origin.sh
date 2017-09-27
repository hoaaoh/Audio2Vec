#!/bin/bash -ex 

[ -f path.sh ] && . ./path.sh

if [ $# != 7 ] ; then 
  echo "usage: train_origin.sh <feat_dir> <hidden_dim> <model_dir> <log_dir> <GPU DEVICE> <MAX TRAINING STEP>"
  echo "e.g. train_origin.sh ./feat 400 ./model ./logs 1 80000"
  echo "The feat dir should conatin:"
  echo "cmvned_feats.ark all.scp all_prons words.txt"
fi 

feat_dir=$1
dim=$2
model_dir=$3/$dim
log_dir=$4/$dim
export CUDA_VISIBLE_DEVICES=$5
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

mkdir -p $model_dir $log_dir $tf_model_dir $tf_log_dir

[ -f $feat_dir/cmvned_feats.ark ] || exit 1
[ -f $feat_dir/all_prons ] || exit 1

if [ ! -d $feat_dir/feats ]; then
  mkdir -p $feat_dir/feats
  get_feat.py $feat_dir/all_prons $feat_dir/cmvned_feats.ark $feat_dir/feats \
		2> $log_dir/get_feat.log
fi

tf_feat_dir=$feat_dir/tf_form
mkdir -p $tf_feat_dir
if [ ! -d $feat_dir ];then
  test_num=0
  [ -f $feat_dir/all_AE.scp ] && rm $feat_dir/all_AE.scp
  [ -f $feat_dir/all_NE.scp ] && rm $feat_dir/all_NE.scp
  for file in $feat_dir/feats/50/*
  do
    outname=$(basename "$file")
    dir_name=$(dirname "$file")
    kaldi_to_tfrecords.py $file $tf_feat_dir/$outname 2>> $log_dir/tfrecord.log
    echo "$tf_feat_dir/$outname" >> $feat_dir/all_AE.scp
    single_num=`wc -l $file | cut -f1 -d ' '`
    test_num=$((test_num + single_num))
    echo $dir_name/$outname >> $1
  done 
fi

### split train/(test -> query/corpus) sets ###
tmp=$((mktemp))

if [ ! -f $feat_dir/train_AE.scp ] && [ ! -f $feat_dir/query_AE.scp ] && [ ! -f $feat_dir/corpus_AE.scp ] ;
then
  tail -n 3 $feat_dir/all_AE.scp > $tmp
  head -n=-3 $feat_dir/all_AE.scp > $feat_dir/train_AE.scp
  cat $tmp | head -n 1 > $feat_dir/query_AE.scp 
  cat $tmp | tail -n 2 > $feat_dir/corpus_AE.scp
  rm $tmp
fi

if [ ! -f $feat_dir/train_NE.scp ] && [ ! -f $feat_dir/query_NE.scp ] && [ ! -f $feat_dir/corpus_NE.scp ] ;
then 
  tail -n 3 $feat_dir/all_NE.scp > $tmp
  head -n=-3 $feat_dir/all_NE.scp > $feat_dir/train_NE.scp
  cat $tmp | head -n 1 > $feat_dir/query_NE.scp 
  cat $tmp | tail -n 2 > $feat_dir/corpus_NE.scp
  rm $tmp
fi 



### training ###
#export CUDA_VISIBLE_DEVICES=1
audio2vec_train.py --init_lr=1 --decay_rate=500 --hidden_dim=$dim --max_step=$6 \
  $tf_log_dir $tf_model_dir $feat_dir/train_AE.scp 2> $tf_log_dir/train.log

### echo $dim > $tf_model_dir/$feat_dim/feat_dim

