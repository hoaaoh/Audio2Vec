#!/usr/bin/env python3

import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import argparse 
FLAG = None




def main():
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='do the PCA transform and plot the output')
    parser.add_argument('train_file',
        help='the training file with feats and labels')
    parser.add_argument('word_dic', 
        help='the dictionary of the label')
    parser.add_argument('--target_words',
        help='main PCA works on only the specific words')
    FLAG = parser.parse_args()


    main()


