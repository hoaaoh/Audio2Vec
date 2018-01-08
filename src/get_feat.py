#!/usr/bin/env python3
import os
import argparse

FLAG = None


def read_classify_list(filename):
    classify_dic = {}

    with open(filename, 'r') as f:
        for line in f:
            l_sp = line.rstrip().split(' ')
            ID = l_sp[0]
            start_frame = l_sp[1]
            cont_frame = l_sp[2]
            word_id = int(l_sp[3])
            if ID not in classify_dic:
                classify_dic[ID] = []
            classify_dic[ID].append((start_frame, cont_frame, word_id))

    return classify_dic


def mkdir(frame_num_list, path):
    for i in frame_num_list:
        if not os.path.exists(path+'/'+str(i)):
            os.mkdir(path+'/'+str(i))
    return


def classify(frame_num, frame_num_list):
    for i in range(len(frame_num_list)-1):
        if i == 0 and frame_num < frame_num_list[i]:
            return frame_num_list[i]
        if frame_num < frame_num_list[i+1] and frame_num >= frame_num_list[i]:
            return frame_num_list[i+1]
    return frame_num_list[-1]


# write the features to path/classify_num.ark ###
def read_and_save_feat(filename, classify_dic, frame_num_list, path, feat_dim):
    import numpy as np
    counter_dic = {}
    for i in frame_num_list:
        counter_dic[i] = 0
    with open(filename, 'r') as f:
        for line in f:
            if '[' in line:
                ID = line.strip().split(' ')[0]
                # temp_list contains all the utterance feature ###
                temp_list = []
                for lines in f:
                    flag = False
                    if ']' in lines:
                        lines = lines.replace(']', ' ')
                        flag = True
                    feat_str = lines.strip().split(' ')
                    feat_l = [float(i) for i in feat_str]
                    temp_list.append(feat_l)
                    if flag:
                        break
                if ID not in classify_dic:
                    continue
                for start, cont, word_id in classify_dic[ID]:
                    if word_id == 0:
                        continue
                    cls = classify(int(cont), frame_num_list)
                    # if cls != 50 :
                    #    continue
                    new_frames = [temp_list[i] for i in range(int(start),
                                  int(start)+int(cont))]
                    # padding zero ###
                    # print (cont, cls)

                    if int(cont) < cls:
                        new_frames += [[0. for j in range(feat_dim)]
                                       for i in range(cls - int(cont))]
                        np_new_frames = np.reshape(np.array(new_frames), -1)
                        np_new_frames = np.append(np_new_frames, [word_id])
                    # print (np_new_frames[0])

                    with open(path + '/' + str(cls) + '/' +
                              str(int(counter_dic[cls]/FLAG.num_in_ark)) +
                              '.ark', 'a') as csvfile:
                        counter_dic[cls] += 1
                        for i in range(len(np_new_frames)):
                            if i != len(np_new_frames)-1:
                                csvfile.write(str(np_new_frames[i])+',')
                            else:
                                csvfile.write(str(np_new_frames[i])+'\n')
    return


def main():
    classify_list = [100, 200]
    path = FLAG.store_path
    mkdir(classify_list, path)
    classify_dic = read_classify_list(FLAG.prons)
    read_and_save_feat(FLAG.feat_ark,
                       classify_dic,
                       classify_list,
                       path,
                       FLAG.feat_dim)
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
    parser.add_argument('--feat_dim',
                        type=int,
                        default=39,
                        help='the feat dimension, default=39')
    parser.add_argument('--num_in_ark',
                        type=int,
                        default=2000,
                        help='the number of lines in each ark file')
    FLAG = parser.parse_args()
    main()
