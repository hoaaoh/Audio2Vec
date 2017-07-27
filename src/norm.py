#!/usr/bin/env python3

import argparse 
import numpy as np
import data_parser as dp
FLAG = None
def get_mean_var(feats):
    np_target = np.array(feats)
    mean = np.mean(np_target,0)
    var = np.var(np_target,0)
    
    return mean, var

def normalize(feats, mean, var):
    np_feats = np.array(feats)
    np_feats = np_feats - mean
    feats = np_feats.tolist()
    
    return feats


def parse_opt():
    parser = argparse.ArgumentParser(
        description='To do the normalize of query and corpus files')
    parser.add_argument('query_fn',
        metavar='<query filename>',
        help='The filename of query file with feat and label')
    parser.add_argument('corpus_fn',
        metavar='<the database filename>',
        help='The database filename with feat and label')
    parser.add_argument('--test_num',type=int,default=100,
        metavar='--test number',
        help='The testing number for MAP')
    return parser



def main():
    test_feats, test_labs = dp.read_csv_file(FLAG.query_fn)
    train_feats, train_labs = dp.read_csv_file(FLAG.corpus_fn)
    if len(test_feats[0]) != len(train_feats[0]):
        print (len(test_feats[0]), len(train_feats[0]))
        raise NameError('The dimension between two files are not the same')
    feat_dim = len(test_feats[0])
    mean, var = get_mean_var(train_feats)
    train_normed_feats = normalize(train_feats, mean, var)
    test_normed_feats = normalize(test_feats, mean, var)
    dp.write_feat_lab(FLAG.query_fn +'_normed', test_normed_feats, test_labs)
    dp.write_feat_lab(FLAG.corpus_fn +'_normed', train_normed_feats, train_labs)
    #print (MAP(test_list[:FLAG.test_num],train_list, feat_dim=feat_dim))

    return 

if __name__ == '__main__':

    parser = parse_opt()
    FLAG = parser.parse_args()
    main()

