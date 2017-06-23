#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 2 ]; then
  echo "usage: script/get_DTW_scp.sh <DTW_DIR> <output_scp>"
  echo "e.g.:  script/get_DTW_scp.sh ./DTW_feats ./DTW.scp"
  exit 1;
fi

DTW_DIR=$1
cur_dir=`pwd`
dir=$cur_dir/$1

ALL=$DTW_DIR/*

[ -d $2 ] && rm $2

for FILE in $ALL
do
  fn=$(basename "$FILE")
  echo $dir/$fn >> $2  
done

