#!/usr/bin/env python3

import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
# import MAP_eval as MAP
import data_parser as reader
import argparse 
import PCA_eval as PCA

FLAG = None
color_list = ['firebrick','red','darkorange','orange','forestgreen','lime',
    'aqua', 'dodgerblue','orchid', 'darkmagenta']

def TSNE_2d_plotting(feat_trans, ax):
    x = []
    y = []
    for i in feat_trans:
        x.append(i[0])
        y.append(i[1])
        
    ax.scatter(x,y)
    return ax 

def main():
    ### preprocessing ###
    label_type = 'words'
    if FLAG.trans_file[-4:] == 'spks':
        label_type = 'spks'
    feats, labs = reader.read_csv_file(FLAG.trans_file, label_type)
    dic, rev_dic = reader.build_dic(FLAG.word_dic, label_type)
    targets = reader.build_targets(FLAG.target_words, dic)
    test_feat_dic = PCA.extract_targets(feats, labs, targets)
    word_color_dict = reader.build_label_color_list(targets, color_list)
    
    ave_test_feat_dic = PCA.average_over_words(test_feat_dic)
    ave_test_feat_list = [ ave_test_feat_dic[i] for i in ave_test_feat_dic]
    test_all_feats, test_all_delta_labs = PCA.target_dic2list(test_feat_dic)
    print (len(test_all_feats))

    # ave_num_feats, ave_num_lab = PCA.average_over_words_num(test_feat_dic, targets)
    ave_num_feats, ave_num_lab = test_all_feats, test_all_delta_labs
    print (len(ave_num_feats))
    #ave_num_feats, ave_num_lab = PCA.target_dic2list(ave_num_feat_lists)
    #sampled_feats, sampled_delta_lab = PCA.sampling(test_all_feats,
    #    test_all_delta_labs)
    sampled_feats, _ = PCA.PCA_transform(ave_num_feats)
    ave_test_feat_trans = PCA.TSNE_transform(sampled_feats, FLAG.tsne_dim)
    print (len(ave_test_feat_trans))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = PCA.plot_all_color(ave_test_feat_trans, ave_num_lab, rev_dic, ax, 
        word_color_dict)
    #ax = PCA.plot_with_anno( ave_test_feat_trans, anno_list, rev_dic, ax)
    #ax.legend(loc='upper right')
    # plt.show()
    plt.savefig(FLAG.trans_file + '.png')

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='do the TSNE transform and plot the output')
    parser.add_argument('trans_file',
        help='the trasforming file with feats and labels')
    parser.add_argument('word_dic',
        help='the dictionary of the label')
    parser.add_argument('target_words',
        help='main TSNE works on only the specific words')
    parser.add_argument('--other_words',type=str, default='None',
        help='apply the TSNE on the words not for training and then plot')
    parser.add_argument('--ave_num',type=int, default=10,
        help='the number for averaging over the word occurance and '
             'get the average feat vectors')
    parser.add_argument('--sample_num',type=int, default=10,
        help='the number for sampling while plotting, '
             'default=10')
    parser.add_argument('--tsne_dim',type=int,default=2,
        help='the dimension to project the transformed TSNE features. '
             'default=2')
    FLAG = parser.parse_args()
    PCA.FLAG = FLAG
    PCA.FLAG.pca_dim = 20

    main()


