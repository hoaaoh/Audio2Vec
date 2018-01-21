#!/bin/bash

[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Audio2Vec
#feat_dir=/home_local/grtzsohalf/yeeee/French
feat_dir=/nfs/Mazu/grtzsohalf/yeeee/German

gram_num=2

[ -f $feat_dir/cmvned_feats.ark ] || exit 1
[ -f $feat_dir/all_prons ] || exit 1
mkdir -p $feat_dir/feats

if [ ! -f $feat_dir/feats/extracted ]; then
  [ -f $feat_dir/feats ] && rm -rf $feat_dir/feats
  [ -f $feat_dir/filtered_prons ] && rm -rf $feat_dir/filtered_prons
  mkdir -p $feat_dir/feats
  python3 $path/src/get_feat.py --gram_num=$gram_num $feat_dir/all_prons $feat_dir/cmvned_feats.ark \
    $feat_dir/feats $feat_dir/filtered_prons 2> get_feat.log
  echo 1 > $feat_dir/feats/extracted
fi
