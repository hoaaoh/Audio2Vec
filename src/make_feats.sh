#!/bin/bash

[ -f path.sh ] && . ./path.sh
path=/home/grtzsohalf/Audio2Vec
feat_dir=/home_local/grtzsohalf/yeeee

[ -f $feat_dir/cmvned_feats.ark ] || exit 1
[ -f $feat_dir/all_prons ] || exit 1
mkdir -p $feat_dir/feats

if [ ! -f $feat_dir/feats/extracted ]; then
  [ -f $feat_dir/feats ] && rm -rf $feat_dir/feats
  mkdir -p $feat_dir/feats
  python get_feat.py --num_in_ark=10000 $feat_dir/all_prons $feat_dir/cmvned_feats.ark $feat_dir/feats \
		2> get_feat.log
  echo 1 > $feat_dir/feats/extracted
fi
