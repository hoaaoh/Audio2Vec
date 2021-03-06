#!/usr/bin/env python3 
import numpy as np
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import random
import argparse

FLAG = None
#work_dir = "/home_local/hoa/Research/Interspeech2017/baseline"
def AP(query_embed, all_embed, answer_inds, id_num, feat_dim=100):
    '''Computes average precision of one single query 
    args:
      query_embed: the embedding of one single query
      all_embed: all embeddings in the database
      answer_inds: correct indexes 
      id_num: the query order in database
    return:
      ave_P: average precision
    '''
    pair_list = []
    answer = []

    np_query = np.array(query_embed).reshape(1,feat_dim)
    np_all = np.array(all_embed).reshape(-1,feat_dim)
    answer = cosine_similarity(np_query, np_all).flatten()
    # answer = euclidean_distances(np_query, np_all).flatten()

    total_right = 0
    for i, answer_i in enumerate(answer):
        if i == id_num :
            continue
        pair_list.append((answer_inds[i],answer_i))
        
        if answer_inds[i] == 1:
            total_right += 1

    pair_list = sorted(pair_list, key=itemgetter(1), reverse=True)
    # Calculate P@R (R-Precision)  
    ave_P = 0.
    right = 0.
    total_right = min(10,total_right)
    for i in range(total_right):
        if pair_list[i][0] == 1:
            right += 1
            ave_P += right/(i+1) 
    
    if total_right==0:
        ave_P = -1
    else:
        ave_P /= total_right
    return ave_P

def transform_answer_index(all_list, target_answer):
    answer_inds = []
    for word_id in all_list:
        if word_id == target_answer :
            answer_inds.append(1)
        else:
            answer_inds.append(0)
    return  answer_inds


def MAP(db_embed_label, feat_dim=100):
    ''' Computes mean average precision (MAP) 
    args:
      test_embed: the query embedding
      all_embed: all doc embeddings
    return:
      MAP: the value of MAP
    '''

    MAP = 0.
    db_embed = []
    db_label = []
    zero_count= 0
    for i, embed_label in enumerate(db_embed_label):
        db_embed.append(embed_label[0])
        db_label.append(embed_label[1])

    
    for i, single_embed in enumerate(db_embed_label):
        ans_inds = transform_answer_index(db_label, single_embed[1])
        ap = AP(single_embed[0],db_embed,ans_inds, i ,feat_dim)
        if ap == -1:
            zero_count+=1
        else:
            MAP += ap
    MAP /= len(db_embed_label) -zero_count

    return MAP

def split_train_test(all_list):
    sampled_test = sorted(random.sample(range(len(all_list)),1000))
    test_list = []
    train_list = []
    # print (sampled_test)
    cnt = 0 
    for i in range(len(all_list)):
        if cnt < 1000 and i == sampled_test[cnt] :
            test_list.append(all_list[i])
            cnt += 1 
        else: 
            train_list.append(all_list[i])
    return train_list, test_list

def write_list(object_list, filename):
    with open(filename,'w') as f:
        for embed, word_id in object_list:
            for one_dim in embed:
                f.write(str(one_dim)+' ')
            f.write(str(word_id) + '\n')
    return 

def read_list(filename):
    return_list = []
    with open(filename) as f:
        for line in f:
            line_sp = line.rstrip().split(' ')
            embed = list(map(float,line_sp[:-1]))
            word_id = int(line_sp[-1])
            return_list.append([embed,word_id])
    return return_list 

def main():
    train_list = read_list(FLAG.db_fn)
    '''
    if len(test_list[0][0]) != len(train_list[0][0]):
        print (len(test_list[0][0]), len(train_list[0][0]))
        raise NameError('The dimension between two files are not the same')
    '''
    feat_dim = len(train_list[0][0])

    print (MAP(train_list, feat_dim=feat_dim))

    return 

def parse_opt():
    parser = argparse.ArgumentParser(
        description='To evaluate the MAP score between two files with vectors.')
    parser.add_argument('db_fn',
        metavar='<the database filename>',
        help='The database filename with feat and label')
    parser.add_argument('--test_num',type=int,default=100,
        metavar='--test number',
        help='The testing number for MAP')
    return parser

if __name__ == '__main__' :
    parser=parse_opt()
    FLAG=parser.parse_args()

    main()


