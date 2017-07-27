#!/usr/bin/env python3

import argparse
import data_parser as dp
import numpy as np
from scipy import spatial
FLAG = None



def edit_distance(list1, list2):
    '''
    args:
      list1: a list with string elements 
      list2: a list with string elements
        list1 and list2 should have the same type
        
    return:
      compute_mat[-1][-1]: the edit distance of two
        different list

    '''
    compute_mat = [[0 for i in range(len(list1)+1) ]
        for j in range(len(list2)+1)]
    for i in range(len(list1)+1):
        compute_mat[0][i] = i
    for i in range(len(list2)+1):
        compute_mat[i][0] = i
    for i in range(1,len(list2)+1):
        for j in range(1,len(list1)+1):
            if list1[j-1] == list2[i-1] :
                compute_mat[i][j] = compute_mat[i-1][j-1]
            else:
                compute_mat[i][j] = min( 
                    1+ compute_mat[i-1][j],    # insertion
                    1 + compute_mat[i][j-1],   # deletion
                    1 + compute_mat[i-1][j-1]) # substitution

    return compute_mat[-1][-1]



def gen_similarity_average(q_feats,q_labs, d_feats, d_labs, lex_dic):
    '''
    Args:
      q_feats: the query feature list, with shape (feat_num, feat_dim)
      q_labs: the query label list, with length = feat_num
      d_feats: the data feature list
      d_labs: the data label list

    Returns:
      bucks: with 5 different phoneme edit distance average cosine similarity
    '''

    bucks = [ 0. for i in range(10)]
    bucks_var = [[] for i in range(10)]
    bucks_cnt  =[ 0 for i in range(10)]
    right_cnt = 0
    for i, q_lab in enumerate(q_labs):
        for j, d_lab in enumerate(d_labs):
            E_D = edit_distance(lex_dic[q_lab], lex_dic[d_lab])
            if E_D < 10:
                #right_cnt +=1 
                #print (E_D, lex_dic[q_lab], lex_dic[d_lab])
                bucks_cnt[E_D] += 1
                bucks[E_D] += 1 - spatial.distance.cosine(q_feats[i],
                    d_feats[j])
                bucks_var[E_D].append(1- spatial.distance.cosine(q_feats[i],
                    d_feats[j]))
    for i in range(len(bucks)):
        if bucks_cnt[i] == 0 :
            print("bucket " + str(i) + "has no occurance")
            break
        bucks[i] /= bucks_cnt[i]
    for i in bucks_var:
        print (np.mean(np.array(i),0), np.var(np.array(i),0))
        
    return bucks, bucks_cnt

def main():
    word_dic,word_rev_dic = dp.build_dic(FLAG.word_dic)
    query_feats, query_labs = dp.read_csv_file(FLAG.test_fn)
    data_feats, data_labs = dp.read_csv_file(FLAG.train_fn)

    #tmp = np.array(query_feats + data_feats)
    #tmp_p = tmp -np.mean(np.array(tmp),0)
    #tmp_list = tmp_p.tolist()
    #query_feats = tmp_list[:len(query_feats)]
    #data_feats = tmp_list[len(query_feats):]
    #dp.write_feat_lab(FLAG.test_fn + '_normed', query_feats, query_labs)
    #dp.write_feat_lab(FLAG.train_fn + '_normed', data_feats, data_labs)
    #query_feats = query_feats[:100]
    #query_labs = query_labs[:100]
    #data_feats = data_feats[:10000]
    #data_labs = data_labs[:10000]

    lex_dic =  dp.build_lexicon(FLAG.lexicon, word_dic)
    bucks, bucks_cnt = gen_similarity_average(query_feats, query_labs, 
        data_feats, data_labs, lex_dic)
    print (bucks, bucks_cnt)
    return


def parse_opt():
    parser = argparse.ArgumentParser(
        description='Generate the cosine similarity mean of '
        'edit distance from one to five.')
    parser.add_argument('lexicon',
        metavar='<lexicon.txt>',
        help='The lexicon file')
    parser.add_argument('test_fn',
        metavar='<testing filename>',
        help='The testing filename')
    parser.add_argument('train_fn',
        metavar='<training filename>',
        help='The training filename')
    parser.add_argument('word_dic',
        metavar='<word dictionary>',
        help='The word dictionary')


    return parser 

if __name__ == '__main__':
    parser = parse_opt()
    FLAG = parser.parse_args()
    main()
