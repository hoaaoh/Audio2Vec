#!/usr/bin/env python3

import numpy as np 
import random
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import MAP_eval 
fn = "/home_local/hoa/Research/Interspeech2017/baseline/words/word_files"
word_dir = "/home_local/hoa/Research/Interspeech2017/baseline/words/"
word_list = []
word_dic = {}
reverse_dic = {}

with open(word_dir+"words.txt") as f:
    for line in f:
        word, word_id = line.rstrip().split(' ')
        word_dic[int(word_id)] = word
        reverse_dic[word] =  int(word_id)

with open(fn) as f:
    for line in f:
        word_list.append(int(line.rstrip()))

X = []
label = []
for word_id in word_list:
    with open(word_dir+str(word_id)) as f:
        for line in f:
            label.append(word_id)
            dim_str = line.rstrip().split(',')
            X_single = list(map(float, dim_str))
            X.append(X_single)

print (len(X))
X = np.array(X)
pca = PCA(n_components=2, svd_solver="arpack")
pca50 = PCA(n_components=50)
X_trans = pca.fit_transform(X)
X_trans50 = pca50.fit_transform(X)

def get_variance(trans, label):
    cur_lab = label[0]
    variance_list = []
    start = 0
    cnt = 0 
    for i, lab in enumerate(label):
        cnt += 1
        if cur_lab != lab:
            variance_list.append(np.var(trans[start:start+cnt]))
            start = i
            cnt = 0
    return variance_list

def find_word_dist(target_word, all_list, label, rev_dic): 
    w_id = reverse_dic[target_word] 
    start = 0
    for i, lab in enumerate(label):
        if lab == w_id :
            start = i
            break
    cnt = 0 
    while label[start+cnt] == w_id:
        cnt += 1
    return all_list[start:start+cnt]



# average of same words # 
dic = {}
for i, lab in enumerate(label):
    if lab not in dic:
        dic[lab] = [0., 0., 0.]
    dic[lab][0] += X_trans[i,0]
    dic[lab][1] += X_trans[i,1]
    dic[lab][2] += 1

for lab in dic:
    dic[lab][0] /= dic[lab][2]
    dic[lab][1] /= dic[lab][2]

# done average # 
y = []
z = []
lab_arr= []
for i, lab in enumerate(dic):
    y.append(dic[lab][0])
    z.append(dic[lab][1])
    lab_arr.append(lab)

print (len(dic))
rand_ind = random.sample(range(len(dic)), 100)
y_rand = []
z_rand = []
lab_rand = []

'''
for rand in rand_ind:
    y_rand.append(y[rand])
    z_rand.append(z[rand])
    lab_rand.append(lab_arr[rand])
'''
test_word_fn  =  word_dir + "../src/test_words"
testing_words = []
with open(test_word_fn) as f:
    for line in f:
        testing_words.append( line.rstrip())

for i in testing_words:
    word_id = reverse_dic[i]
    y_rand.append(dic[word_id][0])
    z_rand.append(dic[word_id][1])
#    print (i + " appears " + str(dic[word_id][2]) +" times")
# y = X_trans[:10000,0]
# z = X_trans[:10000,1]

def plot_fig_with_anno(x_dim, y_dim, anno_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_dim, y_dim)
    for i, w in enumerate(anno_list):
        ax.annotate(w, (x_dim[i],y_dim[i]))
    plt.grid()
    plt.show()
    return 

def plot_fig(x_dim, y_dim):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_dim, y_dim)
    plt.grid()
    plt.show()
    return 

tmp_list = find_word_dist("DOGS",X_trans, label, reverse_dic)
yy = []
zz = []
for i in range(len(tmp_list)):
    yy.append(tmp_list[i][0])
    zz.append(tmp_list[i][1])

plot_fig(yy,zz)


# all_list = []
# for i, lab in enumerate( label):
#     all_list.append([X[i], lab])
# train_list, test_list = MAP_eval.split_train_test(all_list)
# MAP_eval.write_list(train_list, word_dir + "/train_list")
# MAP_eval.write_list(test_list, word_dir+ "/test_list")


def main():


    return 




