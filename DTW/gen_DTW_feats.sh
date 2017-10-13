#!/bin/bash -ex

[ -f /home_local/hoa/libri_feat/query_NE.scp ] || exit 1
[ -f /home_local/hoa/libri_feat/corpus_NE.scp ] || exit 1
[ -f path.sh ] && . path.sh
feat_dir=/home_local/hoa/libri_feat
query=$feat_dir/query_NE.scp 
corpus=$feat_dir/corpus_NE.scp
nfs_dir=/nfs/Mazu
query_feat_dir=$nfs_dir/hoa/query_DTW
corpus_feat_dir=$nfs_dir/hoa/corpus_DTW

query_flist=$nfs_dir/hoa/query_DTW.scp
corpus_flist=$nfs_dir/hoa/corpus_DTW.scp
[ -f $query_flist ] && rm $query_flist
[ -f $corpus_flist ] && rm $corpus_flist
[ -d $query_feat_dir ] && rm -rf $query_feat_dir
mkdir $query_feat_dir
[ -d $corpus_feat_dir ] && rm -rf $corpus_feat_dir 
mkdir $corpus_feat_dir

cnt=0
while read line; do
  lc=`wc -l $line | cut -f 1 -d ' '`
  trans_DTW_feats.py --feat_dim=39 $line $query_feat_dir $query_flist $cnt
  cnt=$((cnt+lc))

done < $query

cnt=0
while read line; do
  lc=`wc -l $line | cut -f 1 -d ' '`
  trans_DTW_feats.py --feat_dim=39 $line $corpus_feat_dir $corpus_flist $cnt
  cnt=$((cnt+lc))
done < $corpus

