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
gradient_flip=1.0
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee
dim=$1
#model_dir=$path/exp/model_lr${init_lr}_gf${gradient_flip}/$dim
#log_dir=$path/exp/log_lr${init_lr}_gf${gradient_flip}/$dim
model_dir=$path/exp/model_lr${init_lr}_hinge/$dim
log_dir=$path/exp/log_lr${init_lr}_hinge/$dim
tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log
device_id=$2
max_step=$3

[ -f $feat_dir/cmvned_feats.ark ] || exit 1
[ -f $feat_dir/all_prons ] || exit 1
mkdir -p $feat_dir/feats
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir
if [ ! -f $feat_dir/feats/extracted ]; then

  [ -f $feat_dir/feats ] && rm -rf $feat_dir/feats
  mkdir -p $feat_dir/feats
  get_feat.py --num_in_ark=10000 $feat_dir/all_prons $feat_dir/cmvned_feats.ark $feat_dir/feats \
		2> $log_dir/get_feat.log
  echo 1 > $feat_dir/feats/extracted
fi

tf_feat_dir=$feat_dir/tf_form

mkdir -p $tf_feat_dir
if [ ! -f $feat_dir/extracted ];then
  test_num=0
  [ -f $feat_dir/all_AE.scp ] && rm $feat_dir/all_AE.scp
  [ -f $feat_dir/all_NE.scp ] && rm $feat_dir/all_NE.scp
  # file_count=$(ls -lR $feat_dir/feats/50 | wc -l)
  # echo $file_count
  # done_count=$((0))
  # file_list=($feat_dir/feats/50/*)
  # for file in $feat_dir/feats/50/*
  num_tfrecords=100
  for ((i=0;i<${num_tfrecords};++i))
  do
    # outname=$(basename "$file")
    # dir_name=$(dirname "$file")
    # echo ${file_list[RANDOM % ${#file_list[@]}]} 
    #kaldi_to_tfrecords.py ${file_list[RANDOM % ${#file_list[@]}]} $feat_dir/feats/50 $file \
    #   $tf_feat_dir/$outname 2>> $log_dir/tfrecord.log
    kaldi_to_tfrecords.py $num_tfrecords ${i} $feat_dir/feats/50 $tf_feat_dir/${i}.ark # 2>> $log_dir/tfrecord.log
    echo "$tf_feat_dir/$i.ark" >> $feat_dir/all_AE.scp
    # single_num=`wc -l $feat_dir/feats/50/${i}.ark | cut -f1 -d ' '`
    # test_num=$((test_num + single_num))
    # echo $feat_dir/feats/50/${i}.ark >> $feat_dir/all_NE.scp
    echo 1 > $feat_dir/extracted
    # done_count=$(($done_count + 1))
    # echo $done_count
  done 
fi

### split train/(test -> query/corpus) sets ###
tmp=$(mktemp)

#if [ ! -f $feat_dir/train_AE.scp ] && [ ! -f $feat_dir/query_AE.scp ] && [ ! -f $feat_dir/corpus_AE.scp ] ;
if [ ! -f $feat_dir/train_AE.scp ] && [ ! -f $feat_dir/test_AE.scp ] ;
then
#   tail -n 1 $feat_dir/all_AE.scp > $tmp
#   head -n -1 $feat_dir/all_AE.scp > $feat_dir/train_AE.scp
  head -n -10 $feat_dir/all_AE.scp > $feat_dir/train_AE.scp
  tail -n 10 $feat_dir/all_AE.scp > $feat_dir/test_AE.scp
  # cat $tmp | head -n 5 > $feat_dir/query_AE.scp 
  # cat $tmp | tail -n 25 > $feat_dir/corpus_AE.scp
  #rm $tmp
fi

# if [ ! -f $feat_dir/train_NE.scp ] && [ ! -f $feat_dir/query_NE.scp ] && [ ! -f $feat_dir/corpus_NE.scp ] ;
# then 
#   tail -n 30 $feat_dir/all_NE.scp > $tmp
#   head -n -30 $feat_dir/all_NE.scp > $feat_dir/train_NE.scp
#   cat $tmp | head -n 5 > $feat_dir/query_NE.scp 
#   cat $tmp | tail -n 25 > $feat_dir/corpus_NE.scp
#   rm $tmp
# fi 

### training ###
export CUDA_VISIBLE_DEVICES=$device_id
#$path/src/audio2vec_train.py --init_lr=$init_lr --decay_rate=500 --hidden_dim=$dim --max_step=$max_step \
  #$tf_log_dir $tf_model_dir $feat_dir/train_AE.scp 2> $tf_log_dir/train.log

#[ -d $feat_dir/words_AE_query_$dim ] && rm -rf $feat_dir/words_AE_query_$dim
#[ -d $feat_dir/words_AE_corpus_$dim ] && rm -rf $feat_dir/words_AE_corpus_$dim

#mkdir $feat_dir/words_AE_query_$dim
#mkdir $feat_dir/words_AE_corpus_$dim
mkdir $feat_dir/words_AE_test_$dim
mkdir $feat_dir/utters_AE_test_$dim
### evaluation ###
#$path/src/audio2vec_eval.py --dim=$dim --test_num=50000 \
  #$tf_model_dir $tf_log_dir $feat_dir/query_AE.scp $feat_dir/words_AE_query_$dim
#$path/src/audio2vec_eval.py --dim=$dim --test_num=250000 \
  #$tf_model_dir $tf_log_dir $feat_dir/corpus_AE.scp $feat_dir/words_AE_corpus_$dim
$path/src/audio2vec_eval.py --dim=$dim --test_num=254790 \
  $tf_model_dir $tf_log_dir $feat_dir/test_AE.scp \
  $feat_dir/words_AE_test_$dim $feat_dir/utters_AE_test_$dim

#$path/src/trans_dir_to_file.py $feat_dir/words_AE_query_$dim $feat_dir/query_AE_$dim
#$path/src/trans_dir_to_file.py $feat_dir/words_AE_corpus_$dim $feat_dir/corpus_AE_$dim
$path/src/trans_dir_to_file.py $feat_dir/words_AE_test_$dim $feat_dir/test_AE_words_$dim
$path/src/trans_dir_to_file.py $feat_dir/utters_AE_test_$dim $feat_dir/test_AE_utters_$dim

#$path/src/MAP_eval.py --test_num=50000 $feat_dir/query_AE_$dim $feat_dir/corpus_AE_$dim > $feat_dir/MAP_AE_"$dim"_RESULT
### done evaluation ###


