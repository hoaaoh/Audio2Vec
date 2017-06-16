#!/usr/bin/env python3

import argparse 
import data_parser as dp
import gen_len as gl
FLAG = None

def write_DTW_feat(fn, single_feat, single_len):
    s_form = 'LT:{:d},LF:{:d},dT:{:f}'
    dim = FLAG.feat_dim
    with open(fn,'w') as f:
        f.write(s_form.format(single_len,dim,0.1))
        f.write('\n')
        for i in range(single_len):
            for j in range(dim-1):
                f.write('{:f} '.format(single_feat[i*dim +j]))
            f.write('{:f}\n'.format(single_feat[i*dim+dim-1]))

    return 


def transform(fn, dic):
    feats, labs = dp.read_csv_file(fn, ',')
    dirname = FLAG.out_dir
    lens = gl.gen_len(feats)
    for i, lab in enumerate(labs):
        if lab not in dic:
            dic[lab] = 0
        outname = dirname + '/' + str(lab) + '_' + str(dic[lab])
        write_DTW_feat(outname, feats[i], lens[i])
        dic[lab] += 1

    return 

def main():
    label_dic = {}
    with open(FLAG.scp, 'r') as f:
        for line in f:
            transform(line.rstrip(), label_dic)

    return 

def parse_opt():
    parser = argparse.ArgumentParser(
        description='translate the feature into DTW feature.')
    parser.add_argument('scp',
        metavar='<feature scp>',
        help='The feature scp file')
    parser.add_argument('out_dir',
        metavar='<output directory>',
        help='The output directory')
    parser.add_argument('--feat_dim',type=int,default=39,
        metavar='<--dimension>',
        help='The dimension of the feature')
    return parser 

if __name__ == '__main__':
    parser = parse_opt()
    FLAG = parser.parse_args()
    gl.FLAG = FLAG  
    
    main()

