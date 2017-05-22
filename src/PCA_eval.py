#!/usr/bin/env python3

import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import argparse 
import data_parser as reader
FLAG = None
color_list = ['grey','red','sienna','gold','c','greenyellow', 'b',
    'crimson','midnightblue', 'saddlebrown']

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
        dic[i] = [0. for k in range(feat_dim)]
        for feat in feat_dic[i]:
            for k in range(len(feat)):
                dic[i][k] += feat[k]
        for k in range(feat_dim):
            dic[i][k] /= len(feat_dic[i])

    return dic 

def PCA_transform(feats):
    pca = PCA(n_components=2)
    pca.fit(feats)
    feat_trans = pca.transform(feats)
    return feat_trans, pca

def plot_with_anno(f_2d, list_dic, rev_dic, ax):
    x = []
    y = []
    for i, f in enumerate(f_2d):
        x.append(f[0])
        y.append(f[1])

    ax.scatter(x, y, color='k')
    for i in range(len(list_dic)):
        ax.annotate(rev_dic[list_dic[i]], (x[i],y[i]))

    return ax

def plot_all_color(f_2d, delta_lab , ax):
    x = []
    y = []
    for i, f in enumerate(f_2d):
        x.append(f[0])
        y.append(f[1])

    start = 0
    for i in range(len(delta_lab)):
        delta, label = delta_lab[i]
        ax.scatter(x[start:start+delta],y[start:start+delta],
            color=color_list[i%len(color_list)] )

    return  ax


def extract_targets(all_feats, all_labs, targets):
    target_feat_dic = {}
    for ID in targets:
        target_feat_dic[ID] = []
    for i, ID in enumerate(all_labs):
        if ID in targets:
            target_feat_dic[ID].append(all_feats[i])

    return target_feat_dic

def target_dic2list(feat_dic):
    feat_list = []
    delta_lab_list = []
    for i in feat_dic:
        num_occur = len(feat_dic[i])
        for j in feat_dic[i]:
            feat_list.append(j)
        delta_lab_list.append((num_occur,i))

    return feat_list, delta_lab_list

def main():
    ### preprocessing ###
    feats, labs = reader.read_csv_file(FLAG.train_file)
    dic, rev_dic = reader.build_dic(FLAG.word_dic)
    targets = reader.build_targets(FLAG.target_words, dic)
    test_feat_dic = extract_targets(feats, labs, targets)
    
    ### PCA through the average target words ###
    ave_test_feat_dic = average_over_words(test_feat_dic)
    ave_test_feat_list = [ ave_test_feat_dic[i] for i in ave_test_feat_dic]
    anno_list = [ i for i in ave_test_feat_dic ]
    ave_test_trans, model = PCA_transform(ave_test_feat_list)

    ### start plotting the average results ###
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plot_with_anno(ave_test_trans, anno_list, rev_dic, ax)

    ### use the PCA model to transform all data ###
    all_feats, delta_lab_list = target_dic2list(test_feat_dic)
    all_feats_trans = model.transform(all_feats)
    ax = plot_all_color(all_feats_trans, delta_lab_list, ax)
    
    plt.legend('upper left')
    plt.grid()
    plt.show()
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


