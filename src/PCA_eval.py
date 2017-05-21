#!/usr/bin/env python3

import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import argparse 
import data_parser as reader
FLAG = None


def average_over_words(feat_dic):
    '''Assume feat_dic[word_ID] is a list of word utterance
      args:
        feat_dic: a dictionary with 
                  feat_dic[word_ID] = shape(num_occur, feat_dim)
      returns:
        dic: a dict with dic[word_ID] = average over the word utterance.

    '''
    dic= {}
    
    for i in feat_dic:
        feat_dim = len(feat_dic[i][0])
        dic[i] = [0. for k in range(fet_dim)]
        for feat in feat_dic[i]:
            for k in range(len(feat)):
                dic[i][k] += feat[k]
        for k in range(feat_dim):
            dic[i][k] /= len(feat_dic[i])

    return dic 

def PCA_transform(feats):
    pca = PCA(n_components=2)
    feat_trans = pca.fit_transform(feats)

    return feat_trans

def plot_with_anno(f_2d, list_dic):
    x = [], y=[]
    for i, f in enumerate(f_2d):
        x.append(f[0])
        y.append(f[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x, y)

    plt.grid()
    plt.show()

    return 


def extract_targets(all_feats, all_labs, targets):
    target_feat_dic = {}
    for ID in targets:
        target_feat_dic[ID] = []
    for i, ID in enumerate(all_labs):
        if ID in targets:
            target_feat_dic[ID].append(all_feats[i])

    return target_feat_dic


def main():
    feats, labs = reader.read_csv_file(FLAG.train_file)
    dic, rev_dic = reader.build_dic(FLAG.word_dic)
    targets = reader.build_targets(FLAG.target_words)
    test_feat_dic = extract_targets(feats, labs, targets)
    ave_test_feat_dic = average_over_words(test_feat_dic)
    ave_test_feat_list = [ ave_test_feat_dic[i] for i in ave_test_feat_dic]
    anno_list = [ i for i in ave_test_feat_dic ]
    ave_test_trans = PCA_transform(ave_test_feat_list)
    plot_with_anno(ave_test_trans, anno_list)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='do the PCA transform and plot the output')
    parser.add_argument('train_file',
        help='the training file with feats and labels')
    parser.add_argument('word_dic', 
        help='the dictionary of the label')
    parser.add_argument('target_words',
        help='main PCA works on only the specific words')
    FLAG = parser.parse_args()

    main()


