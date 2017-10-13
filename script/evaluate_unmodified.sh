#!/bin/bash  -ex

if [ $# != 4 ]; then 
  echo "Evaluation of Autoencoder & Naive Encoder & DTW "
  echo "usage: script/evaluate_unmodified.sh  <scp dir> <model dir> <log dir>  <output dir> "
  echo "e.g.: script/evaluate_unmodified.sh /hoa/German /hoa/exp/model /hoa/exp/log /hoa/result "
  exit 1;
fi

[ -f path.sh ] . ./path.sh

scp_dir=$1
model_dir=$2
log_dir=$3
result_dir=$4

#### SPOKEN TERM DETECTION USING NAIVE ENCODER        ####
#### Assume using total_NE.scp as DTW query and corpus ###

[ -f $scp_dir/total_NE.scp ] || exit 1
[ -d $result_dir/total_NE ] && rm $result_dir/total_NE/*
mkdir -p $result_dir/total_NE


Naive.py --part_num=10 $scp_dir/total_NE.scp $result_dir/total_NE
trans_dir_to_file.py $result_dir/total_NE $result_dir/NE_tmp
MAP_1ofN.py $result_dir/NE_tmp > $result_dir/NE_MAP_score 
rm $result_dir/NE_tmp

#### SPOKEN TERM DETECTION USING AUDIO2VEC ENCODER    ####
#### Assume using total_AE.scp as DTW query and corpus ###

[ -f $scp_dir/total_AE.scp ] || exit 1
[ -d $result_dir/total_AE ] && rm $result_dir/total_AE/*
mkdir -p $result_dir/total_AE
audio2vec_eval.py --test_num=21000 --dim=400 $model_dir  $log_dir $scp_dir/total_AE.scp   $result_dir/total_AE
trans_dir_to_file.py $result_dir/total_AE $result_dir/AE_tmp
MAP_1ofN.py $result_dir/AE_tmp > $result_dir/AE_MAP_score
rm $result_dir/AE_tmp

#### SPOKEN TERM DETECTION USING Dynamic Time Warping  ####
#### Assume using total_DTW.scp as DTW query and corpus ###

# [ -f $scp_dir/total_DTW.scp ] || exit 1
[ -d $result_dir/total_DTW ] && rm $result_dir/total_DTW/*
mkdir -p $result_dir/total_DTW
cnt=0
while read line; do
  lc=`wc -l $line | cut -f 1 -d ' '`
  trans_DTW_feats.py --feat_dim=39 $line $result_dir/total_DTW $result_dir/total_DTW.scp $cnt
  cnt=$((cnt+lc))
done < $scp_dir/total_NE.scp
example3 $result_dir/total_DTW.scp > $result_dir/DTW_MAP_score
