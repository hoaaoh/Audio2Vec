#!/usr/bin/env python3

import csv 
import argparse


FLAG = None

def write_file(feats,lab_list, fn):
    with open(fn,'w') as f:
        for num, i in enumerate(feats):
            for j in range(len(i)):
                f.write(str(i[j]) + ',')
            f.write(str([len(i)-1]) + '\n')

    return 

def transform(feats, lens):
    dim = FLAG.feat_dim
    trans_feats = [] 

    for i in range(len(feats)):
        trans_feats.append(feats[i][:single_len[lens[i]*dim]])

    return trans_feats

def read_feat(fn):
    feats = []
    labs = []
    with open(fn,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            feats.append(list(map(float,row[:-1])))
            labs.append(float(row[-1]))

    return feats, labs

def read_len(fn):
    len_list = []
    with open(fn,'r') as f:
        for line in f:
            len_list.append(int(line.rstrip()))

    return len_list

def main():
    len_list = read_len(FLAG.len_file)
    ark_list, lab_list = read_feat(FLAG.ark_file)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transfrom the fulfilled zeros to no')
    parser.add_argument('--feat_dim',type=int, default=39,
        help='each frame feat dimension')
    parser.add_argument('ark_file',
        help='the transforming ark file')
    parser.add_argument('len_file',
        help='meaning the length of each utterance')
    parser.add_argument('out_ark',
        help='the output file')
    FLAG = parser.parse_args()

    main()

