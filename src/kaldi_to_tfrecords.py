#!/usr/bin/env python3

import argparse
import os
import sys
import random
from tqdm import tqdm
import time

import numpy as np

FLAGS = None

def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(values):
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _float_feature_list(values):
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def main(unused_argv):

    file_list = os.listdir(FLAGS.feats_dir)
    num_file = len(file_list)
    file_per_tfrecord = int(num_file/int(FLAGS.num_tfrecords))
    print ('files per tfrecord: ' + str(file_per_tfrecord))
    tfrecord_id = int(FLAGS.tfrecord_id)
    start_id = tfrecord_id*file_per_tfrecord
    if (tfrecord_id+1)*file_per_tfrecord > num_file:
        end_id = num_file
    else:
        end_id = (tfrecord_id+1)*file_per_tfrecord 
    print ('start id: ' + str(start_id))
    print ('end id: ' + str(end_id))
    count = 0
    feats_dic = {}
    for file in file_list[start_id: end_id]:
        with open(os.path.join(FLAGS.feats_dir, file), 'r') as f:
            feats_dic[file] = {}
            feats_dic[file]['feats'] = []
            feats_dic[file]['labels'] = []
            feats_dic[file]['utterances'] = []
            line_num = 0
            for line in f:
                line = line.rstrip().split(',')
                if len(line) == 0:
                    print ("len(line)=0")
                    print (file)
                    print (line_num)
                count += 1
                line_num += 1
                feats_dic[file]['feats'].append(list(map(float, line[:-1])))
                feats_dic[file]['labels'].append([int(float(line[-1]))])
                feats_dic[file]['utterances'].append([str.encode(file)])
            feats_dic[file]['len'] = line_num
            if line_num <= 1:
                print (file)
    writer = tf.python_io.TFRecordWriter(FLAGS.output)
    for i in tqdm(range(start_id, end_id)):
        for feat, label, utterance in zip(feats_dic[file_list[i]]['feats'], \
                                          feats_dic[file_list[i]]['labels'], feats_dic[file_list[i]]['utterances']): 
            # time_start = time.time()
            feat_concat = []
            label_concat = []
            utterance_concat = []
            feat_pos = None
            label_pos = None
            utterance_pos = None
            feat_neg = None
            label_neg = None
            utterance_neg = None

            # positive pair
            if feats_dic[file_list[i]]['len'] == 0:
                print (file_list[i])
                continue
            index = random.choice(range(feats_dic[file_list[i]]['len']))
            feat_pos = feats_dic[file_list[i]]['feats'][index]
            label_pos = feats_dic[file_list[i]]['labels'][index]
            utterance_pos = feats_dic[file_list[i]]['utterances'][index]
            # negative pair
            neg_range = list(range(start_id, end_id))
            neg_range.remove(i)
            index_neg = random.choice(neg_range)
            feats_neg = feats_dic[file_list[index_neg]]['feats']
            labels_neg = feats_dic[file_list[index_neg]]['labels']
            utterances_neg = feats_dic[file_list[index_neg]]['utterances']
            if feats_dic[file_list[index_neg]]['len'] == 0:
                print (file_list[index_neg])
                continue
            index = random.choice(range(feats_dic[file_list[index_neg]]['len']))
            feat_neg = feats_neg[index]
            label_neg = labels_neg[index]
            utterance_neg = utterances_neg[index]

            feat_concat.extend(feat)
            feat_concat.extend(feat_pos)
            feat_concat.extend(feat_neg)
            label_concat.extend(label)
            label_concat.extend(label_pos)
            label_concat.extend(label_neg)
            utterance_concat.extend(utterance)
            utterance_concat.extend(utterance_pos)
            utterance_concat.extend(utterance_neg)
            if len(feat_concat) != 5850:
                print ('Q_Q')
                continue
            example = tf.train.Example(features=tf.train.Features(feature={
                'feat': _float_feature(feat_concat),
                'label': _int64_feature(label_concat),
                'utterance': _bytes_feature(utterance_concat)
            }))
            writer.write(example.SerializeToString())
    writer.close()
    print ("word # of " + FLAGS.output + ": " + str(count))

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description = 
         'transform text format kaldi features and labels into tfrecords')

   parser.add_argument(
        'num_tfrecords',
        metavar='<num tfrecords>',
        type=str,
        help='number of tfrecords'
        )
   parser.add_argument(
        'tfrecord_id',
        metavar='<tfrecord id>',
        type=str,
        help='id of tfrecord'
        )
   parser.add_argument(
        'feats_dir',
        metavar='<feats dir>',
        type=str,
        help='feats dir'
        )
   parser.add_argument(
        'output',
        metavar='<output-tf-record>',
        type=str,
        help='generated tf-Records file'
        )
   parser.add_argument(
        '--feats_dim',
        metavar='<feats-dim>',
        type=int,
        default=39,
        help='acoustic feature dimension'
        )
   parser.add_argument(
        '--norm_var',
        metavar='<True|False>',
        type=bool,
        default=False,
        help='Normalize Variance of each sentence'
        )
   parser.add_argument(
        '--norm_mean',
        metavar='<True|False>',
        type=bool,
        default=False,
        help='Normalize mean of each sentence'
        )
   FLAGS, unparsed = parser.parse_known_args()

   import tensorflow as tf
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
