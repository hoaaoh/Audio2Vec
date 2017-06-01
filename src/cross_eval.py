#!/usr/bin/env python3

import PCA_eval as PCA
import MAP_eval as MAP
import data_parser as reader
import argparse 
from  sklearn.externals import joblib

FLAG = None

color_list = ['firebrick','red','darkorange','orange','forestgreen','lime',
    'aqua', 'dodgerblue','orchid', 'darkmagenta']

def main():

    q_feats, q_labs = reader.read_csv_file(FLAG.q_fn)
    db_feats, db_labs = reader.read_csv_file(FLAG.db_fn)
    dic, rev_dic = reader.build_dic(FLAG.word_dic)
    targets = reader.build_targets(FLAG.target_words)
    word_color_dict = reader.build_label_color_list(targets, color_list)
    model = joblib.load(FLAG.model_fn)

    proj_feat_dic = PCA.extract_targets(db_feats, db_labs, targets)

    ### get the target words average ###
    ave_target_feat_dic = PCA.average_over_words(proj_feat_dic)
    ave_target_feat_list = [ ave_test_feat_dic[i] for i in ave_target_feat_dic]
    ave_num_target_feat_list = PCA.average_over_words_num(proj_feat_dic, targets)
    
    anno_list = [i for i in ave_target_feat_dic ]
    ave_target_trans = model.transform(ave_target_feat_list)

    all_feats, delta_lab_list = PCA.target_dic2list(proj_feat_dic)
    all_feats_trans = model.transform(all_feats)

    sampled_feats, sampled_delta_lab = MAP.sampling(all_feats_trans,
        delta_lab_list)

    fig = plt.figure()
    if FLAG.pca_dim == 2:
        ### start plotting the average results ###
        ax = fig.add_subplot(111)
        ax = PCA.plot_with_anno(ave_test_trans, anno_list, rev_dic, ax)
        ### plotting all word utterance ###
        ax = PCA.plot_all_color(sampled_feats, sampled_delta_lab, rev_dic, ax,
            word_color_dict)
        
    elif FLAG.pca_dim ==3 :
        #### start plotting the 3D projections #### 
        ax = fig.add_subplot(111, projection='3d')
        ax = PCA.plot_with_anno_3d(ave_test_trans, anno_list, rev_dic, ax)
        ax = PCA.plot_all_color_3d(sampled_feats, sampled_delta_lab, rev_dic, ax,
            word_color_dict)

    else:
        print ("no plotting but testing through MAP")

        all_list = []
        feat_trans = model.transform(feats)
        print (len(feat_trans[0]))
        for i in range(len(feats)):
            all_list.append((feat_trans[i],labs[i]))
        
        train_list, test_list = MAP.split_train_test(all_list)
        print (MAP.MAP(test_list[:100], train_list, feat_dim=FLAG.pca_dim))
        
        return 

    ax.legend(loc='upper right')
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='using stored model to evaluate '
        'cross lingual domain')
    parser.add_argument('q_fn',
        help='the file that contains the queries feats and labels')
    parser.add_argument('db_fn',
        help='the file that contains the database feats and labels')
    parser.add_argument('word_dic',
        help='the dictionary of the label')
    parser.add_argument('target_words',
        help='to project only the target words and show')
    parser.add_argument('model_fn',
        help='the model name to save/load')
    parser.add_argument('--pca_dim', type=int, default=2,
        help='the dimension to project the transformed features'
             'default=2')


    FLAG = parser.parse_args()

    main()





