#!/usr/bin/env python3

import numpy as np
import data_parser as DP
import argparse 
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



FLAG = None



def cal_sim():
    
    
    return

def main():
    ### preprocessing ###
    src_feats, src_labs = DP.read_csv_file(FLAG.src_file)
    pca_model = joblib.load(FLAG.model)
    ### output transformed features ###
    src_trans_feats = pca_model.transform(src_feats)
    DP.write_feat_lab(FLAG.target_file, src_trans_feats, src_labs)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using to generate the PCA '
        'transformed feature files')
    parser.add_argument('src_file',
        help='the file with features to be transformed')
    parser.add_argument('target_file',
        help='the output file with transformed features')
    parser.add_argument('model',
        help='the PCA model to be loaded')
    FLAG = parser.parse_args()
    main()
