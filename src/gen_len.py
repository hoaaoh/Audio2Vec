#!/usr/bin/env python3

import csv 
import argparse

FLAG = None

def read_csv(fn):
    feats = []
    labs = []
    with open(fn,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            feats.append(list(map(float,row[:-1])))
            labs.append(float(row[-1]))

    return feats, labs
def gen_len(csvlist):
    ''' generate the length of the mfcc list 
    args:
      csvlist: the csv list read out from the csv file 
    returns:
      length_list: the list that records the length of each mfcc occurance
    '''
    dim = FLAG.feat_dim
    length = int(len(csvlist[0])/dim)
    len_list = []
    for i in csvlist:
        each_len = length
        for j in range(length-1,-1,-1):
            all_zero = True
            for k in range(dim):
                if i[j*dim +k] != 0:
                    all_zero = False
                    break
            if not all_zero:
                len_list.append(each_len)
                break
            each_len -= 1
    return len_list

def write_file(len_list, out_fn):
    with open(out_fn, 'w') as f:
        for i in len_list[:-1]:
            f.write(str(i)+'\n')
        f.write(str(len_list[-1]))
    return 

def main():
    feats, labs = read_csv(FLAG.csvfile)
    len_list = gen_len(feats)
    write_file(len_list, FLAG.out_fn)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='generate the length list of each utturance')
    parser.add_argument('--feat_dim',type=int, default=39,
        help='the mfcc dimension')
    parser.add_argument('csvfile',
        help='the file to be parsed')
    parser.add_argument('out_fn',
        help='the output filename')
    FLAG = parser.parse_args()
    main()
