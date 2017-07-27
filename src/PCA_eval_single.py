#!/usr/bin/env python3

import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.externals import joblib 
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from math import floor 
import argparse 
import MAP_eval as MAP
import data_parser as reader


FLAG = None
color_list = ['firebrick','lightsalmon','forestgreen','lime']
another_color_list = ['aqua', 'dodgerblue','orchid',
    'darkmagenta','black','gray']
color_list.extend(another_color_list)
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
        if len(feat_dic[i]) ==0:
            print (str(i) + ' not in corpus')
            continue
        feat_dim = len(feat_dic[i][0])
        dic[i] = [0. for k in range(feat_dim)]
        for feat in feat_dic[i]:
            for k in range(len(feat)):
                dic[i][k] += feat[k]
        for k in range(feat_dim):
            dic[i][k] /= len(feat_dic[i])

    return dic 

def PCA_transform(feats):
    ''' Process PCA to transform feat 
    args:
      feat: feature list, shape = (feat_num, feat_dim)

    returns:
      feat_trans: the transformed feature list,
        shape = (feat_num, pca_project_dim)
      pca: the trained pca model
    '''

    pca = PCA(n_components=FLAG.pca_dim,svd_solver='full')
    pca.fit(feats)
    feat_trans = pca.transform(feats)
    return feat_trans, pca

def plot_with_anno(f_2d, list_dic, rev_dic, ax, marker, lab):
    ''' Plot 2d figurewith annotation
    args:
      f_2d: feature 2d list, shape = (feat_num, 2)
      list_dic: the list of annotations
      rev_dic: the reverse dictionary with rev_dic[ID]=word
      ax: the subplot of plt
    returns:
      ax: the updated subplot 
    '''
    x = []
    y = []
    for i, f in enumerate(f_2d):
        x.append(f[0])
        y.append(f[1])

    ax.scatter(x, y, color='k',marker=marker, label=lab)
    for i in range(len(list_dic)):
        ax.annotate(rev_dic[list_dic[i]], (x[i]+0.01,y[i]+0.01))
        print (x[i],y[i], rev_dic[list_dic[i]])
    return ax

def plot_with_anno_3d(f_3d, list_dic, rev_dic, ax):
    ''' Plot 3d figure with annotation
    args:
      f_3d: feature 2d list, shape = (feat_num, 2)
      list_dic: the list of annotations
      rev_dic: the reverse dictionary with rev_dic[ID]=word
      ax: the subplot of plt
    returns:
      ax: the updated subplot 
    '''
    x = []
    y = []
    z = []
    for i, f in enumerate(f_3d):
        x.append(f[0])
        y.append(f[1])
        z.append(f[2])
    ax.scatter(x, y, z, color='k', label="average_words")
    for i in range(len(list_dic)):
        ax.text(x[i],y[i],z[i],rev_dic[list_dic[i]])
    return ax 


def plot_all_color(f_2d, delta_lab, rev_dic, ax, word_color_dict):
    ''' 2d plotting 
    args:
      f_2d: the 3d features  
      delta_lab: the delta range and the label 
      rev_dic: reverse dictionary, dic[ID] = word
      ax: the subplot of the main plot
      word_color_dict: a dictionary with 
           key = word, value = color
    returns:
      ax: the updated subplot 
    '''

    x = []
    y = []
    for i, f in enumerate(f_2d):
        x.append(f[0])
        y.append(f[1])

    start = 0
    for i in range(len(delta_lab)):
        delta, label = delta_lab[i]
        word_label = rev_dic[label]
        print (word_label)
        print ("x")
        print (x[start:start+delta])
        print ("y")
        print (y[start:start+delta])
        ax.scatter(x[start:start+delta],y[start:start+delta],
            color=word_color_dict[label], label=word_label )
        start += delta
    
    return  ax

def plot_all_color_3d(f_3d, delta_lab, rev_dic, ax, word_color_dict):
    ''' 3d plotting 
    args:
      f_3d: the 3d features  
      delta_lab: the delta range and the label 
      rev_dic: reverse dictionary, dic[ID] = word
      ax: the subplot of the main plot
      word_color_dict: a dictionary with 
           key = word, value = color
    returns:
      ax: the updated subplot 
    '''

    x = []
    y = []
    z = []
    for i, f in enumerate(f_3d):
        x.append(f[0])
        y.append(f[1])
        z.append(f[2])
    start = 0
    for i in range(len(delta_lab)):
        delta, label = delta_lab[i]
        word_label = rev_dic[label]
        ax.scatter(x[start:start+delta], y[start:start+delta],
            z[start:start+delta],
            color=word_color_dict[label],
            label=word_label)
        start += delta
    return ax 

def sampling(f,delta_lab):
    ''' sample through the same label 
    args:
      f: the feats of a word utterance, shape = [num_occur, feat_dim]
      delta_lab: It's a list that combines the label and the occurance time

    returns:
      new_f, new_delta_lab
    '''
    new_f = []
    new_delta_lab = []
    sample_num = FLAG.sample_num

    start = 0 
    for i in delta_lab:
        delta = i[0]
        lab = i[1]
        if delta <= sample_num :
            new_f.extend(f[start:start+delta])
            new_delta_lab.append((delta,lab))
        else:
            tmp_f = random.sample(range(delta), sample_num)
            for i in tmp_f:
                new_f.append(f[start+i])
            new_delta_lab.append((sample_num, lab))
        start += delta
        
    return new_f, new_delta_lab


def extract_targets(all_feats, all_labs, targets):
    ''' extract the target feats in all feats
    args:
      all_feats: 
      all_labs:  
      targets:
    returns:
      target_feat_dic:
    '''
    target_feat_dic = {}
    for ID in targets:
        target_feat_dic[ID] = []
    for i, ID in enumerate(all_labs):
        if ID in targets:
            target_feat_dic[ID].append(all_feats[i])

    return target_feat_dic

def target_dic2list(feat_dic):
    ''' Transform the feat_dic to target 
    args:
      feat_dic: feat_dic[word_ID] = feat lists, shape = (num_occur, feat_dim)
      
    returns:
      feat_list: feat_list with shape [ total_feat_num, feat_dim ] 
      delta_lab_list: the range of delta and the label list
    '''
    feat_list = []
    delta_lab_list = []
    for i in feat_dic:
        num_occur = len(feat_dic[i])
        for j in feat_dic[i]:
            feat_list.append(j)
        delta_lab_list.append((num_occur,i))

    return feat_list, delta_lab_list

def average_over_words_num(feat_dic, target_list):
    ''' given the FLAG.ave_num, average over every FLAG.ave_num occurance 
        of words and returns the feat list
    args:
      feat_dic: feat_dic[word_ID] = feat lists, shape = (num_occur, feat_dim)
      target_list: the target word lists

    retruns:
      feat_lists: averaged 2-dimension matrix, [feat_num, feat_dim] 
      delta_lab_list: the range of delta and the label list
    '''
    num = FLAG.ave_num
    feat_lists = []
    delta_lab_list = []
    
    for i in target_list:
        iter_num = int(floor(float(len(feat_dic[i]))/num))
        if i not in feat_dic:
            print (str(i) + " not in corpus")
            continue
        feat_dim = len(feat_dic[i][0])
        delta = 0
        for j in range(min(30,iter_num)):
            l = [0. for tmp in range(feat_dim)]
            for k in range(feat_dim):
                l[k] += feat_dic[i][j][k]/num
            feat_lists.append(l)
            delta += 1
        delta_lab_list.append((delta,i))
    return feat_lists, delta_lab_list

def extract_additional_words(feat_dic,word_list ):
    '''given the FLAG.ave_num, average over every FLAG.ave_num occurance
       and each word contains only one avearage vector
    args:
      feat_dic: feat_dic[word_ID] = lists of feats, shape=(num_occur, feat_dim)
      word_list: the plotting word list

    returns:
      ave_feat: average feat of target word 
      lb: label list each feat 
    '''
    ave_feat = [] ; lb = []
    for i in word_list:
        num_occur = min(FLAG.ave_num, len(feat_dic[i]))
        print (len(feat_dic[i]))
        feat_dim = len(feat_dic[i][0])
        feat_list = [ 0. for i in range(feat_dim)]
        for j in range(num_occur):
            for k in range(feat_dim):
                feat_list[k] += feat_dic[i][j][k]

        for k in range(feat_dim):
            feat_list[k] /= num_occur
        
        ave_feat.append(feat_list)
        lb.append(i) 
        
    return ave_feat, lb

def plot_additional_words(ave_ftrans, lb, rev_dic,ax):
    '''plot the additional words, i.e. not used for transforming words
    args:
      ave_ftrans_lb_color: the average feature list, 
        each object in the list contains: 
          ([ feat_dim ], label(word_id), color of the word)
      ax: the subplot of the figure 

    returns:
      ax: the updated subplot
    '''
    x = []
    y = []
    
    for i in ave_ftrans:
        x.append(i[0])
        y.append(i[1])
    ax.scatter(x, y, color='b', label='additional words')
    for i in range(len(lb)):
        ax.annotate(rev_dic[lb[i]], (x[i],y[i]))

    return ax

def plot_additional_words_3d(ave_ftrans, lb, rev_dic,ax):
    '''plot the additional words, i.e. not used for transforming words
    args:
      ave_ftrans_lb_color: the average feature list, 
        each object in the list contains: 
          ([ feat_dim ], label(word_id), color of the word)
      ax: the subplot of the figure 

    returns:
      ax: the updated subplot
    '''
    x = []
    y = []
    z = []
    
    for i in ave_ftrans:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    ax.scatter(x, y, z, color='b', label='additional words')
    for i in range(len(lb)):
        ax.text(x[i],y[i],z[i],rev_dic[lb[i]])

    return ax

def TSNE_transform(feats, dim):
    ''' Process TSNE transforming 
    args:
      feats: feature list, shape = (feat_num, feat_dim)
    returns:
      feat_trans: the transformed feature list

    '''
    tsne = TSNE(n_components=dim)
    feat_trans = tsne.fit_transform(feats)

    return feat_trans

def main():
    ### preprocessing ###
    feats, labs = reader.read_csv_file(FLAG.train_file)
    dic, rev_dic = reader.build_dic(FLAG.word_dic)
    targets = reader.build_targets(FLAG.target_words, dic)
    test_feat_dic = extract_targets(feats, labs, targets)
    ave_dic = average_over_words(test_feat_dic)
    ave_feat_list = [ ave_dic[i] for i in ave_dic ] 
    anno_list = [ i for i in ave_dic ]

    for i in test_feat_dic:
        print (i)
        print (len(test_feat_dic[i]))
    #word_color_dict = reader.build_label_color_list(targets, color_list)
    
    another_feats, another_labs=  reader.read_csv_file(FLAG.apply_file)
    another_dic, another_rev_dic=  reader.build_dic(FLAG.apply_dic)
    another_tar = reader.build_targets(FLAG.another_target_words, another_dic)
    another_test_dic = extract_targets(another_feats, another_labs, another_tar)
    another_color_dict = reader.build_label_color_list(another_tar,
        another_color_list)
    test_feats2, test_delta_labs2 = target_dic2list(another_test_dic)
    ave_dic2 = average_over_words(another_test_dic)
    ave_feat_list2 =  [ ave_dic2[i] for i in ave_dic2]
    anno_list2 = [ i for i in ave_dic2 ]
    ### PCA through all average words, eliminating the less occurance of words? ###

    feats_trans, model  = PCA_transform(feats)
    test_feats, test_delta_labs = target_dic2list(test_feat_dic)
    print (len(test_feats))
    test_feats_trans = model.transform(test_feats)
    ### PCA through the average target words ###
    ave_feat_trans_list = model.transform(ave_feat_list)
    #ave_feat_trans_list = tsne.fit_transform(ave_feat_trans_list)
    test_feats2_trans = model.transform(test_feats2)
    ave_feat_trans_list2 = model.transform(ave_feat_list2)
    
    if FLAG.pca_dim == 2:
        fig = plt.figure()
        ### start plotting the average results ###
        ax = fig.add_subplot(111)
        ax = plot_with_anno(ave_feat_trans_list, anno_list, rev_dic, ax, 'o',
        'German')
        ax = plot_with_anno(ave_feat_trans_list2, anno_list2, another_rev_dic,
            ax, 'x', 'French')
        
        ### plotting all word utterance ###
        #ax = plot_all_color(test_feats_trans, test_delta_labs, rev_dic, ax,
        #    word_color_dict)
    #    ax = plot_all_color(test_feats2_trans, test_delta_labs2,
    #        another_rev_dic, ax, another_color_dict)

    elif FLAG.pca_dim ==3 :
        fig = plt.figure()
        #### start plotting the 3D projections #### 
        ax = fig.add_subplot(111, projection='3d')
        # ax = plot_with_anno_3d(ave_test_trans, anno_list, rev_dic, ax)
        ax = plot_all_color_3d(test_feats_trans, test_delta_labs, rev_dic, ax,
            word_color_dict)

    else:
        print ("no plotting but testing through MAP")

        all_list = []
        feat_trans = model.transform(feats)
        # print (len(feat_trans[0]))
        for i in range(len(feats)):
            all_list.append((feat_trans[i],labs[i]))
        
        train_list, test_list = MAP.split_train_test(all_list)
        print (MAP.MAP(test_list[:100], train_list, feat_dim=FLAG.pca_dim))
        
        return 
    ### get the words that not using for PCA ###
    #if FLAG.other_words != 'None':
    #    other_target = reader.build_targets(FLAG.other_words,dic)
    #    extract_others = extract_targets(feats, labs, other_target)
    #    ave_feat, other_lb_list = extract_additional_words(extract_others, other_target)
    #    ave_ftrans  = model.transform(ave_feat)
    #    
    #    if FLAG.pca_dim == 2:
    #        plot_additional_words(ave_ftrans, other_lb_list, rev_dic, ax)
    #    else :
    #        plot_additional_words_3d(ave_ftrans, other_lb_list, rev_dic, ax)

    ax.legend(loc='upper right')
    plt.show()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='do the PCA transform and plot the output')
    parser.add_argument('train_file',
        help='the training file with feats and labels')
    parser.add_argument('apply_file')
    parser.add_argument('apply_dic')
    parser.add_argument('another_target_words')
    parser.add_argument('word_dic', 
        help='the dictionary of the label')
    parser.add_argument('target_words',
        help='main PCA works on only the specific words')
    parser.add_argument('--pca_dim',type=int,default=2,
        help='the dimension to project the transformed PCA features. '
             'default=2')
    FLAG = parser.parse_args()

    main()


