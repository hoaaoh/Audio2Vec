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


