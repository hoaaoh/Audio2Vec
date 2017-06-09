#!/bin/bash -ex

FILES=/home_local/hoa/Research/Interspeech2017/baseline/words/*

for f in $FILES 
do 
	echo $f
	num=`wc -l $f | cut -d ' ' -f 1`
	echo $num
	[ $num -gt 10 ] || rm $f
done 
