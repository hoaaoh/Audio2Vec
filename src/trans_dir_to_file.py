#!/usr/bin/env python3

import os
import glob
import argparse
import random

FLAG = None

def parse_single_file(fn):
    feat_list = []
    with open(FLAG.dir_name + '/' + fn, 'r') as f:
        for line in f:
            line_sp = line.rstrip().split(',')
            line_sp.append(fn)
            feat_list.append(line_sp)

    return feat_list

def get_all_files():
    return_list = []
    for i in os.listdir(FLAG.dir_name):
        if i == '0':
            continue
        return_list.extend(parse_single_file(i))
        
    return return_list

def split_train_test(feat_list):
    shuf = list(range(len(feat_list)))
    random.shuffle(shuf)
    test_list = []
    train_list= []
    for i in range(len(feat_list)):
        if i < FLAG.test_num:
            test_list.append(feat_list[shuf[i]])
        else:
            train_list.append(feat_list[shuf[i]])

    return train_list, test_list

def write_file(fn, feat_list):
    with open(fn,'w') as f:
        for i, feat_lab in enumerate(feat_list):
            for j, feat in enumerate(feat_lab):
                if j != len(feat_lab)-1:
                    f.write(feat+' ')
                else:
                    f.write(feat+'\n')
    return

def main():
    all_feat_list = get_all_files()
    # train, test = split_train_test(all_feat_list)
    write_file(FLAG.out_name, all_feat_list)
    # write_file(FLAG.out_name+'_test', test)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='a python script for dividing whole data into'
        ' training and testing set')
    parser.add_argument('dir_name',
        metavar='<directory name>',
        help='the directory that contains a lot of features in label file')
    parser.add_argument('out_name',
        metavar='<output name>',
        help='the output will be extensioned with train and test\n'
        'eg: tmp_train, tmp_test')
    parser.add_argument('--test_num',type=int, 
        default=1000,
        metavar='<num of testing>',
        help='the number of testing number')
    FLAG = parser.parse_args()
    main()

