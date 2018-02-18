#!/bin/bash

[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Audio2Vec
#feat_dir=/home_local/grtzsohalf/yeeee/French
feat_dir=/nfs/Mazu/grtzsohalf/yeeee/English
feats_dir=/nfs/YueLao/grtzsohalf/yeeee/English/feats

[ -f $feat_dir/cmvned_feats.ark ] || exit 1
[ -f $feat_dir/all_prons ] || exit 1
mkdir -p $feats_dir

if [ ! -f $feats_dir/extracted ]; then
  [ -f $feats_dir ] && rm -rf $feats_dir
  [ -f $feat_dir/filtered_prons ] && rm -rf $feat_dir/filtered_prons
  python3 $path/src/get_feat.py $feat_dir/all_prons $feat_dir/cmvned_feats.ark $feats_dir $feat_dir/filtered_prons
  echo 1 > $feat_dir/feats/extracted
fi
