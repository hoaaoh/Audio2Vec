import sys
import re
import os 
import argparse
from collections import deque

FLAG = None
def read_classify_list(filename):
    classify_dic = {}
    with open(filename,'r') as f:
        for line in f:
            l_sp = line.rstrip().split(' ')
            ID = l_sp[0]
            start_frame = l_sp[1]
            cont_frame = l_sp[2]
            word_id = int(l_sp[3])
            if ID not in classify_dic:
                classify_dic[ID] = []
            classify_dic[ID].append((start_frame,cont_frame,word_id))

    return classify_dic

def mkdir(frame_num_list, path):
    for i in frame_num_list:
        if not os.path.exists(path+'/'+str(i)):
            os.mkdir(path+'/'+str(i))
    return 

def classify(frame_num, frame_num_list):
    for i in range(len(frame_num_list)):
        if i==0 and frame_num <= frame_num_list[i]:
            return frame_num_list[i]
        if frame_num <= frame_num_list[i] and frame_num > frame_num_list[i-1] :
            return frame_num_list[i]
    return None

### write the features to path/classify_num.ark ###
def read_and_save_feat(prons, filename, classify_dic, frame_num_list, path, feat_dim, filtered_prons):
    import csv 
    import numpy as np
    counter_dic = {}
    for i in frame_num_list:
        counter_dic[i] = 0
    filtered_lines = []
    with open(filename,'r') as f:
        line_id = 0
        for line in f:
            if '[' in line:
                # print (line_id)
                ID = line.strip().split(' ')[0]
                
                ### temp_list contains all the utterance feature ###
                temp_list = []
                for lines in f:
                    flag = False
                    if ']' in lines:
                        lines = lines.replace(']',' ')
                        flag = True
                    feat_str = lines.strip().split(' ')
                    feat_l = [float(i) for i in feat_str]
                    temp_list.append(feat_l)
                    if flag :
                        break
                if ID not in classify_dic:
                    continue
                for start, cont, word_id in classify_dic[ID]:
                    if word_id == 0:
                        line_id += 1
                        continue
                    cls = classify(int(cont),frame_num_list)
                    if cls == None:
                        line_id += 1
                        continue
                    filtered_lines.append(line_id)
                    line_id += 1
                    new_frames = [ temp_list[i] for i in \
                        range(int(start),int(start)+int(cont))]
                    ### padding zero ###
                    #print (cont, cls)

                    # if int(cont) < cls : 
                        # new_frames +=  [[0. for j in range(feat_dim)]for i in range(cls -
                            # int(cont))]
                    new_frames = np.array(new_frames, dtype=np.float32)
                    x = np.arange(cls)
                    xp = np.linspace(0, cls - 1, new_frames.shape[0])
                    new_frames = np.vstack([np.interp(x, xp, new_frames[:, i]) for i in \
                                            range(new_frames.shape[1])]).T
                    np_new_frames = np.reshape(np.array(new_frames),-1)
                    np_new_frames = np.append(np_new_frames,[word_id])
                    np_new_frames = np.append(np_new_frames,[ID])
                    #print (np_new_frames[0])
                    
                    # with open(path+'/'+str(cls)+'/'+str(int(counter_dic[cls]/FLAG.num_in_ark)) + '.ark','a') as csvfile:
                    with open(path+'/'+str(cls)+'/'+ID.split('-')[0] + '.ark','a') as csvfile:
                        counter_dic[cls] += 1
                        # csvfile.write(ID+' ')
                        for i in range(len(np_new_frames)):
                            if i != len(np_new_frames)-1:
                                csvfile.write(str(np_new_frames[i])+',')
                            else:
                                csvfile.write(str(np_new_frames[i])+'\n')

    with open(prons, 'r') as fin:
        with open(filtered_prons, 'w') as fout:
            count = 0
            for i, idx in enumerate(filtered_lines):
                while True:
                    line = fin.readline()
                    if line == '':
                        break
                    line = line[:-1]
                    if count == idx:
                        fout.write(line + '\n')
                        count += 1
                        break
                    count += 1

def main():
    
    classify_list = [70]
    path=FLAG.store_path
    mkdir(classify_list, path)
    classify_dic = read_classify_list(FLAG.prons)
    read_and_save_feat(FLAG.prons, FLAG.feat_ark, classify_dic, classify_list, path, FLAG.feat_dim, FLAG.filtered_prons)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get the feat csv through all_prons')
    parser.add_argument('prons',
        help='the pronounciation file')
    parser.add_argument('feat_ark',
        help='the feat ark')
    parser.add_argument('store_path',
        help='the directory to store the feat arks.')
    parser.add_argument('filtered_prons',
        help='the prons file filtered by num of frames')
    parser.add_argument('--feat_dim', type=int,
        default=39,
        help='the feat dimension, default=39')
    
    FLAG = parser.parse_args()


    main()
    
