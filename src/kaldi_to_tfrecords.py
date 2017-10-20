#!/usr/bin/env python3

import argparse
import os
import sys
import random
from tqdm import tqdm

import numpy as np

FLAGS = None

def get_ark(ark_file):
    feats = []
    ### label is word ID ###
    labels = []
    for line in ark_file:
        line = line.rstrip().split(',')
        # print (len(arr_ark))
        # if len(arr_ark) % FLAGS.feats_dim != 1 : return data,0
        try:
            feats.append(map(float, line[0:-1]))
            labels.append([int(float(line[-1]))])
        except ValueError:
            print ("Value Error")
    return feats, labels

def get_ark_and_lab(ark_file):
   # if line == '': return None, None, None

   # arr = line.split()
   # key, data = arr[0],  []

   finish, first = False, True
   for line_ark in ark_file:
      arr_ark = line_ark.split()
      if first:
         first = False
         if arr_ark[0] != key: raise AssertionError()
      else:
         if arr_ark[-1] == ']':
            arr_ark = arr_ark[:-1]
            finish = True

         feats = map(float, arr_ark)
         # if len(feats) != FLAGS.feats_dim: raise AssertionError()

         data.extend(feats)

      if finish: break
   
   if FLAGS.norm_var or FLAGS.norm_mean:
      length = len(data)
      arr = np.array(data)
      arr = arr.reshape(length/FLAGS.feats_dim, FLAGS.feats_dim)

      if FLAGS.norm_mean:
         mean = np.mean(arr, axis = 0)
         arr  = arr - mean
      if FLAGS.norm_var:
         std  = np.std(arr, axis = 0)
         arr  = arr / std

      data = arr.reshape(length).tolist()

   return (key, data)

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
    tfrecord_id = int(FLAGS.tfrecord_id)
    start_id = tfrecord_id*file_per_tfrecord
    if (tfrecord_id+1)*file_per_tfrecord > num_file:
        end_id = num_file
    else:
        end_id = (tfrecord_id+1)*file_per_tfrecord 
    feats_dic = {}
    for file in file_list[start_id: end_id]:
        with open(os.path.join(FLAGS.feats_dir, file), 'r') as f:
            feats_dic[file] = []
            for line in f:
                feats_dic[file].append(line)
                if len(line) == 0:
                    print ("len(line)=0")
                    print (file)
    count = 0
    writer = tf.python_io.TFRecordWriter(FLAGS.output)
    for i in tqdm(range(start_id, end_id)):
        feats = []
        ### label is word ID ###
        labels = []
        utterances = []
        for line in feats_dic[file_list[i]]:
            line = line.rstrip().split(',')
            # print (len(arr_ark))
            # if len(arr_ark) % FLAGS.feats_dim != 1 : return data,0
            try:
                feats.append(list(map(float, line[0:-1])))
                labels.append([int(float(line[-1]))])
                utterances.append([str.encode(file_list[i])])
            except ValueError:
                print ("Value Error")
            count += 1
        # length = len(feats) / FLAGS.feats_dim
        for feat, label, utterance in zip(feats, labels, utterances):  
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
            index = random.choice(range(len(feats)))
            feat_pos = feats[index]
            label_pos = labels[index]
            utterance_pos = utterances[index]
            # negative pair
            # file_neg_name = random.choice(file_list)
            file_neg_name = random.choice(list(feats_dic.keys()))
            feats_neg = []
            ### label is word ID ###
            labels_neg = []
            utterances_neg = []
            for line in feats_dic[file_neg_name]:
                line = line.rstrip().split(',')
                # print (len(arr_ark))
                # if len(arr_ark) % FLAGS.feats_dim != 1 : return data,0
                try:
                    feats_neg.append(list(map(float, line[0:-1])))
                    labels_neg.append([int(float(line[-1]))])
                    utterances_neg.append([str.encode(file_neg_name)])
                except ValueError:
                    print ("Value Error")
            if feats_neg == None or len(feats_neg) == 0:
                print ('@_@')
                continue
            index = random.choice(range(len(feats_neg)))
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
