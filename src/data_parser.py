#!/usr/bin/env python3
import sklearn
import argparse
import numpy as np
import csv

def read_csv_file(filename):
    lab = []
    feats = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            feats.append(list(map(float,row[:-1])))
            lab.append(int(row[-1]))
    return feats, lab

def build_dic(filename):
    dic = {}
    rev_dic = {}
    with open(filename,'r') as f:
        for line in f:
            word, ID= line.rstrip().split()
            dic[word] = int(ID)
            rev_dic[int(ID)] = word

    return dic, rev_dic

def build_targets(filename, dic):
    targets = []
    with open(filename,'r') as f:
        for line in f:
            targets.append(dic[line.rstrip()])

    return targets
