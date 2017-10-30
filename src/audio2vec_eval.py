#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import date
from datetime import datetime 
import math
import tensorflow as tf 
import tempfile
import numpy as np
import time
import argparse


from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.client import timeline
from tensorflow.python.ops import math_ops
import audio2vec_train as audio2vec

log_dir = None
model_dir = None
word_dir= None

batch_size = 500
memory_dim = 1000
seq_len = 50
feat_dim = 39
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 63372


def eval_once(saver, total_loss, summary_writer, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint #
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            # /my-path/model/model.ckpt-0 #
            # extract global_step from it.
            global_step = \
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return 
        # Start queue runners. 
        coord = tf.train.Coordinator()
        try: 
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,\
                                                start=True))
            num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TEST / 
                batch_size))
            step = 0
            total_loss_value = 0.
            while step < num_iter and not coord.should_stop():
                single_loss = sess.run([total_loss])
                print (single_loss)
                total_loss_value += single_loss[0]
                step += 1
            avg_loss = total_loss_value/num_iter
            print ('%s: average loss for eval = %.3f' % (datetime.now(),
                avg_loss))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='eval loss', simple_value=avg_loss)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    return 

def get_BN_feat(saver, reconstruction_loss, summary_writer, summary_op, labels, utterances, s_enc, p_enc):
    """Getting Bottleneck Features"""
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return 
        # Start queue runners. 
        coord = tf.train.Coordinator()
        try: 
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,\
                                                start=True))
            num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TEST / 
                batch_size))
            step = 0
            total_loss_value = 0.
            
            while step < num_iter and not coord.should_stop():
                ### words is a tensor with shape (batch_size, 1) ###
                ### memories are seq_len list with (batch, feat_dim) shape tensors ###
                single_loss, words, utters, s_memories, p_memories = \
                    sess.run([reconstruction_loss, labels, utterances, s_enc, p_enc])
                ### print(np.shape(memories))
                for i in range(len(words)):
                    word_id = words[i]
                    single_memory = p_memories[i]
                    single_memory = single_memory.tolist()
                    with open(word_dir+'/'+str(word_id), 'a') as word_file:
                        for j in range(split_enc, len(single_memory)):
                            word_file.write(str(single_memory[j]))
                            if j != len(single_memory)-1:
                                word_file.write(',')
                            else:
                                word_file.write('\n')
                    with open(utter_dir+'/'+str(utters[i]), 'a') as utter_file:
                        for j in range(split_enc):
                            utter_file.write(str(single_memory[j]))
                            if j != split_enc-1:
                                utter_file.write(',')
                            else:
                                utter_file.write('\n')
                            
                # print (single_loss)
                total_loss_value += single_loss
                step += 1
            avg_loss = total_loss_value/num_iter
            print ('%s: average loss for eval = %.3f' % (datetime.now(),
                avg_loss))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='eval loss', simple_value=avg_loss)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    return 

def BN_evaluation(fn_list):
    """ Getting Bottleneck Features """
    with tf.Graph().as_default() as g:
        examples, labels, utterances = audio2vec.batch_pipeline(fn_list, batch_size,
            feat_dim, seq_len)
        examples = [examples[i] for i in range(seq_len*3) if i%3 == 0]
        labels = labels[0]
        utterances = utterances[0]
        
        W_enc_p = tf.get_variable("enc_w_p", [memory_dim - split_enc, memory_dim - split_enc])
        b_enc_p = tf.get_variable("enc_b_p", shape=[memory_dim - split_enc])
        W_enc_s = tf.get_variable("enc_w_s", [split_enc, split_enc])
        b_enc_s = tf.get_variable("enc_b_s", shape=[split_enc])
        with tf.variable_scope('encoding') as scope_1_1:
            with tf.variable_scope('encoding_p') as scope_1_1_1:
                p_enc = audio2vec.encode(examples, memory_dim - split_enc)
                p_enc = audio2vec.leaky_relu(tf.matmul(p_enc, W_enc_p) + b_enc_p)
            with tf.variable_scope('encoding_s') as scope_1_1_2:
                s_enc = audio2vec.encode(examples, split_enc)
                s_enc = audio2vec.leaky_relu(tf.matmul(s_enc, W_enc_s) + b_enc_s)
        W_dec = tf.get_variable("dec_w", [memory_dim, memory_dim])
        b_dec = tf.get_variable("dec_b", shape=[memory_dim])
        dec_state = audio2vec.leaky_relu(tf.matmul(tf.concat([s_enc,p_enc], 1), W_dec) + b_dec)
        dec_out = audio2vec.decode(examples, batch_size, memory_dim, seq_len, feat_dim, dec_state)

        reconstruction_loss = audio2vec.loss(dec_out, examples, seq_len, batch_size, feat_dim) 
        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir + '/BN', g)
        get_BN_feat(saver, reconstruction_loss, summary_writer, summary_op, labels,
            utterances, s_enc, p_enc)
        return 

def parser_opt():
    parser = argparse.ArgumentParser(
        description='evaluation with seq2seq model')
    parser.add_argument('model_dir',
        metavar='<model directory>',
        help='the model to be evaluated')
    parser.add_argument('log_dir',
        metavar='<log directory>',
        help='the log directory')
    parser.add_argument('file_scp',
        metavar='<file scp>',
        help='the file with feature filenames')
    parser.add_argument('word_dir',
        metavar='<word directory>',
        help='the directory where stores generated bottleneck features')
    parser.add_argument('utter_dir',
        metavar='<utter directory>',
        help='the directory where stores generated bottleneck features')
    parser.add_argument('--dim',type=int,default=400,
        metavar='<hidden layer dimension>',
        help='The hidden layer dimension')
    parser.add_argument('--batch_size',type=int,default=500,
        metavar='<--batch size>',
        help='The batch size of the evaluation')
    parser.add_argument('--test_num',type=int,default=63372,
        metavar='<The testing number of each languages>',
        help='The testing number of each languages')
    parser.add_argument('--split_enc',type=int,default=20,
        metavar='<split enc>',
        help='split enc')
    return parser 
        
def main():
    fn_list = \
    audio2vec.build_filename_list(FLAG.file_scp)
    BN_evaluation(fn_list)
    return 

if __name__ == '__main__':
    parser = parser_opt()
    FLAG = parser.parse_args()
    model_dir = FLAG.model_dir
    log_dir = FLAG.log_dir
    word_dir = FLAG.word_dir
    utter_dir = FLAG.utter_dir
    memory_dim = FLAG.dim
    batch_size = FLAG.batch_size
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST  = FLAG.test_num
    split_enc = FLAG.split_enc
    main()
