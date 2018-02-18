import argparse
import os
import sys
import random
import numpy as np
import time
from tqdm import tqdm

def split_data(feat_dir, exp_dir, n_files, proportion, seq_len):
    feats_dir = os.path.join(feat_dir, 'feats', str(seq_len))
    print(exp_dir)
    train_scp = os.path.join(exp_dir, 'train_AE.scp')
    test_scp = os.path.join(exp_dir, 'test_AE.scp')
    file_list = os.listdir(feats_dir)[:n_files]
    # random.shuffle(file_list)
    n_train_files = int(n_files*proportion)
    print ("Training number of speakers: " + str(n_train_files))
    print ("Testing number of speakers: " + str(n_files - n_train_files))
    with open(train_scp, 'w') as fout:
        for file in file_list[:n_train_files]:
            fout.write(file + '\n')
    with open(test_scp, 'w') as fout:
        for file in file_list[n_train_files:]:
            fout.write(file + '\n')

def write_data_file(feats_dir, scp_file, write_file):
    feats = []
    spk2feat = {}
    feat2label = {}
    spk_list = []
    count = 0
    feats_walk_dir, seq_len = feats_dir.rsplit('/', 1)
    seq_len = int(seq_len)
    with open(scp_file, 'r') as f:
        for line in tqdm(f):
            line = line[:-1]
            print(line)
            spk_list.append(line)
    print(feats_walk_dir)
    x = np.arange(seq_len)
    with open(write_file, 'w') as f:
        f.write("SPK\tCHP\tAUD\tNUM\tWID\tLEN\n")
        for root, dirs, files in os.walk(feats_walk_dir):
            for filename in files:
                if '0.npy' in filename:
                    continue
                spk = filename.split('-')[0] + '.ark'
                if spk in spk_list:
                    sid, chp, aud = filename.split('.')[0].split('-')
                    aud, num, wid = aud.split('_')
                    count = len(feats)
                    feat = np.squeeze(np.load(os.path.join(root, filename))).T
                    f.write('\t'.join([sid, chp, aud, num, wid, str(feat.shape[0])]) + '\n')
                    f.flush()
                    feat_dims = feat.shape[1]

                    y = []
                    xp = np.linspace(0, seq_len - 1, feat.shape[0])
                    for i in range(feat_dims):
                        fp = feat[:, i]
                        y.append(np.interp(x, xp, fp))
                    feat = np.vstack(y).T
                    # if feat.shape[0] < seq_len:
                        # data = np.zeros((seq_len, feat_dims))
                        # data[:feat.shape[0], :feat.shape[1]] = feat
                        # feat = data
                    # else:
                        # feat = feat[:seq_len, :]

                    feats.append(feat.flatten().astype(np.float32))
                    feat2label[count] = (filename.split('_')[-1][:-4], spk)
                    if spk not in spk2feat:
                        spk2feat[spk] = []
                    spk2feat[spk].append(count)
    print ('# of words: ' + str(count))
    return count, feats, spk2feat, feat2label, spk_list

def load_data(feats_dir, scp_file):
    feats = []
    spk2feat = {}
    feat2label = {}
    spk_list = []
    count = 0
    feats_walk_dir, seq_len = feats_dir.rsplit('/', 1)
    seq_len = int(seq_len)
    with open(scp_file, 'r') as f:
        for line in tqdm(f):
            line = line[:-1]
            print(line)
            spk_list.append(line)
    print(feats_walk_dir)
    x = np.arange(seq_len)
    for root, dirs, files in os.walk(feats_walk_dir):
        for filename in files:
            if '0.npy' in filename:
                continue
            spk = filename.split('-')[0] + '.ark'
            if spk in spk_list:
                count = len(feats)
                feat = np.squeeze(np.load(os.path.join(root, filename))).T
                feat_dims = feat.shape[1]

                y = []
                xp = np.linspace(0, seq_len - 1, feat.shape[0])
                for i in range(feat_dims):
                    fp = feat[:, i]
                    y.append(np.interp(x, xp, fp))
                feat = np.vstack(y).T
                # if feat.shape[0] < seq_len:
                    # data = np.zeros((seq_len, feat_dims))
                    # data[:feat.shape[0], :feat.shape[1]] = feat
                    # feat = data
                # else:
                    # feat = feat[:seq_len, :]

                feats.append(feat.flatten().astype(np.float32))
                feat2label[count] = (filename.split('_')[-1][:-4], spk)
                if spk not in spk2feat:
                    spk2feat[spk] = []
                spk2feat[spk].append(count)
    count = len(feats)
    print ('# of words: ' + str(count))
    return count, feats, spk2feat, feat2label, spk_list

def load_mfcc(feats_dir, scp_file):
    feats = []
    spk2feat = {}
    feat2label = {}
    spk_list = []
    count = 0
    _, seq_len = feats_dir.rsplit('/', 1)
    seq_len = int(seq_len)
    x = np.arange(seq_len)
    with open(scp_file, 'r') as fin1:
        for line in tqdm(fin1):
            line = line[:-1]
            spk_list.append(line)
            spk_indices = []
            with open(os.path.join(feats_dir, line), 'r') as fin2:
                for feat in fin2:
                    feat = feat.split(',')
                    data = list(map(float, feat[:-1]))
                    feat_dims = len(data) // seq_len
                    data = np.array(data, dtype=np.float32).reshape((-1, feat_dims))
                    data = data[~np.all(data == 0, axis=1)]
                    print(data.shape)

                    y = []
                    xp = np.linspace(0, seq_len - 1, data.shape[0])
                    for i in range(feat_dims):
                        fp = data[:, i]
                        y.append(np.interp(x, xp, fp))
                    data = np.vstack(y).T

                    feats.append(data)
                    feat2label[count] = (feat[-1][:-1], line)
                    spk_indices.append(count)
                    count += 1
                spk2feat[line] = spk_indices
    print ('# of words: ' + str(count))
    return count, feats, spk2feat, feat2label, spk_list

def batch_pair_data(feats, spk2feat, feat2label, feat_indices, spk_list):
    batch_data = []
    batch_data_pos = []
    batch_data_neg = []
    for idx in feat_indices:
        # feat
        feat = feats[idx]
        word = feat2label[idx][0]
        spk = feat2label[idx][1]

        # feat_pos
        idx_pos = random.choice(spk2feat[spk])
        feat_pos = feats[idx_pos]

        # feat_neg
        spk_list.remove(spk)
        rand_spk = random.choice(spk_list)
        spk_list.append(spk)
        idx_neg = random.choice(spk2feat[rand_spk])
        feat_neg = feats[idx_neg]
        batch_data.append(feat)
        batch_data_pos.append(feat_pos)
        batch_data_neg.append(feat_neg)
        # print (spk + ': (' + str(idx) + ', ' + str(idx_pos) + ') <---> ' + rand_spk + ': ' + str(idx_neg))
    return np.array(batch_data, dtype=np.float32), np.array(batch_data_pos, dtype=np.float32), \
        np.array(batch_data_neg, dtype=np.float32)
