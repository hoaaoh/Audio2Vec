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


def eval_once(saver, total_loss, summary_writer, summary_op, enc_inp, dec_output):
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
                single_loss, inputs, outputs= sess.run([total_loss, enc_inp, dec_output])
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

def get_BN_feat(saver, total_loss, summary_writer, summary_op,labels, dec_memory):
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
                single_loss, words, memories = sess.run([total_loss, labels, dec_memory])
                ### print(np.shape(memories))
                for i in range(len(words)):
                    word_id = words[i][0]
                    single_memory = memories[i]
                    single_memory = single_memory.tolist()
                    with open(word_dir+'/'+str(word_id), 'a') as word_file:
                        for j in range(len(single_memory)):
                            word_file.write(str(single_memory[j]))
                            if j != len(single_memory)-1:
                                word_file.write(',')
                            else:
                                word_file.write('\n')
                            
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
        examples, labels = audio2vec.batch_pipeline(fn_list, batch_size,
            feat_dim, seq_len)
        
        dec_out, enc_memory = audio2vec.inference(examples, batch_size,
            memory_dim, seq_len, feat_dim)
        total_loss = audio2vec.loss(dec_out, examples, seq_len, batch_size,
            feat_dim)
        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir + '/BN', g)
        get_BN_feat(saver, total_loss, summary_writer, summary_op, labels,
            enc_memory)
        return 

def evaluate_and_out(fn_list):
    """ Evaluation """
    with tf.Graph().as_default() as g:
        # Get evaluation sequences #
        examples, labels = audio2vec.batch_pipeline(fn_list, batch_size,
            feat_dim, seq_len)
        # build a graph that computes the results #
        dec_out, enc_memory = audio2vec.inference(examples, batch_size, memory_dim, seq_len,
            feat_dim)
        # calculate loss 
        total_loss = audio2vec.loss(dec_out, examples, seq_len, batch_size, feat_dim)
        
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op =  tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir + '/eval', g)

        #while True:
        eval_once(saver, total_loss, summary_writer, summary_op,examples, dec_out)
        time.sleep(60*5)
        return 


def evaluate(fn_list):
    """ Evaluation """
    with tf.Graph().as_default() as g:
        # Get evaluation sequences #
        examples, labels = audio2vec.batch_pipeline(fn_list, batch_size,
            feat_dim, seq_len)
        # build a graph that computes the results #
        dec_out, enc_memory = audio2vec.inference(examples, batch_size, memory_dim, seq_len,
            feat_dim)
        # calculate loss 
        total_loss = audio2vec.loss(dec_out, examples, seq_len, batch_size, feat_dim)
        
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op =  tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir + '/eval', g)

        #while True:
        eval_once(saver, total_loss, summary_writer, summary_op)
        time.sleep(60*5)
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
    parser.add_argument('--dim',type=int,default=100,
        metavar='<hidden layer dimension>',
        help='The hidden layer dimension')
    parser.add_argument('--batch_size',type=int,default=500,
        metavar='<--batch size>',
        help='The batch size of the evaluation')
    parser.add_argument('--test_num',type=int,default=63372,
        metavar='<The testing number of each languages>',
        help='The testing number of each languages')
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
    memory_dim = FLAG.dim
    batch_size = FLAG.batch_size
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST  = FLAG.test_num
    main()
