#!/usr/bin/env python3
import sklearn
import argparse
import numpy as np
import csv

def read_csv_file(filename, delimiter=' '):
    labs = []
    feats = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            feats.append(list(map(float,row[:-1])))
            labs.append(row[-1])
    return feats, labs

def build_dic(filename):
    dic = {}
    rev_dic = {}
    with open(filename,'r') as f:
        for line in f:
            word, ID= line.rstrip().split()
            dic[word] = ID
            rev_dic[ID] = word

    return dic, rev_dic

def build_targets(filename, dic):
    targets = []
    with open(filename,'r') as f:
        for line in f:
            targets.append(dic[line.rstrip()])

    return targets

def build_label_color_list(target_list, color_list):
    ''' assume the targets are in pairs 


    '''
    word_color_dict = {}
    for i, target in enumerate(target_list):
        word_color_dict[target] = color_list[i%len(color_list)]

    return word_color_dict

def write_feat_lab(filename, feats, labs, delimiter=' '):
    ''' write a csv like file with feature list and label list
    args:
      filename: The output filename
      feats: the feature list with (num_occur, feat_dim)
      labs: the label list with length = num_occur
      delimiter: the delimiter in csv file
    '''
    with open(filename,'w') as f:
        for i, feat_list in enumerate(feats):
            for j in feat_list:
                f.write(str(j)+ delimiter)
            f.write(str(labs[i])+'\n')

    return 

def write_feat_in_lab_name(feat_dic, out_dir, delimiter=' '):
    ''' write a label file with features 
    args:
      feat_dic: The dictionary with dic[lab]=shape(num_occur,feat_dim)
      out_dir : The output directory of the generated features
      delimiter:The csv file delimiter, default=' '
    ''' 
    
    for i in feat_dic:
        with open(out_dir+'/'+str(i),'a') as f:
            for j in feat_dic[i]:
                for k in range(len(j)-1):
                    f.write(str(j[k])+delimiter)
                f.write(str(j[len(j)-1])+ '\n')

    return 

def build_lexicon(fn, word_dic):
    ''' build the lexicon map 
    args:
      fn: the lexicon file name
    return:
      lexicon_dic: the lexicon dictionary
        with lexicon_dic[word_id] = list of phonemes
    '''
    lexicon_dic = {}
    with open(fn,'r') as f:
        for line in f:
            line_sp = line.rstrip().split(' ')
            lexicon_dic[word_dic[line_sp[0]]] = []
            for i in range(1,len(line_sp)):
                lexicon_dic[word_dic[line_sp[0]]].append(line_sp[i])

    return  lexicon_dic
