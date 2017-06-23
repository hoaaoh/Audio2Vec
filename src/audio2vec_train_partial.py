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

FLAG = None

def keep_train(fn_list, batch_size, memory_dim, seq_len=50, feat_dim=39):
    with tf.Graph().as_default():
       # get examples and labels for seq2seq #
        examples, labels = audio2vec.batch_pipeline(fn_list, batch_size, feat_dim, seq_len)

        # build a graph that computes the results
        dec_out, dec_memory = audio2vec.inference(examples, batch_size, memory_dim, seq_len,\
            feat_dim)
        # calculate loss
        total_loss = audio2vec.loss(dec_out, examples, seq_len, batch_size, feat_dim)

        ### learning rate decay ###
        learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("learning rate", learning_rate)

        # build a graph that grains the model with one batch of examples and
        # updates the model parameters
        train_op = audio2vec.train_opt(total_loss, learning_rate, 0.9)
        
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        tf.summary.scalar("RMSE loss", total_loss)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        sess.graph.finalize()
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_writer = tf.summary.FileWriter(log_file,sess.graph)
        # model restoring #
        ckpt = tf.train.get_checkpoint_state(model_file)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = \
              int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            print ('No checkpoint file found.')
        print ("Model restored.")
        print ("Start batch training.")
        feed_lr = FLAG.init_lr
        ### start training ###
        for step in range(0, FLAG.max_step):
            try:
                start_time = time.time()
                _, loss_value = sess.run([train_op, 
                    total_loss], feed_dict={learning_rate:feed_lr})
                
                duration = time.time() - start_time
                example_per_sec = batch_size/duration
                epoch = floor(batch_size * step/FLAG.num_ex)
                format_str = ('%s: epoch %d, step %d, LR %.5f, loss = %.2f ( %.1f examples/sec;'
                    ' %.3f sec/batch)')
                
                print (format_str % (datetime.now(), epoch, step, feed_lr, loss_value,
                    example_per_sec, float(duration)), end='\r')
                
                # create time line #
                #num_examples_per_step = batch_size
                #tl = timeline.Timeline(run_metadata.step_stats)
                #ctf = tl.generate_chrome_trace_format(show_memory=True)
                if step % 200 == 0:
                    ckpt = model_file + '/model.ckpt'
                    summary_str = sess.run(summary_op,feed_dict={learning_rate:
                        feed_lr})
                    saver.save(sess, ckpt, global_step=step)
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()
                    #with open('timeline_'+str(step)+'.json','w') as f:
                    #    f.write(ctf)
                if step % FLAG.decay_rate == FLAG.decay_rate-1 :
                    feed_lr *= FLAG.decay_factor
            except tf.errors.OutOfRangeError:
                break
        coord.request_stop()
        coord.join(threads)
        summary_writer.flush()
    return


def parser_opt():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Keep Training on Other Language Script')
    parser.add_argument('--init_lr',  type=float, default=0.1,
        metavar='<--initial learning rate>')
    parser.add_argument('--num_ex',type=int,default=40000,
        metavar='<--number of examples in the language>')
    parser.add_argument('--max_step',type=int,default=10000,
        metavar='<--number of training epoch with pretrained model>')
    parser.add_argument('--decay_rate',type=int, default=1000,
        metavar='learning rate decay per batch epoch') 
    parser.add_argument('--hidden_dim',type=int, default=100,
        metavar='<--hidden dimension>',
        help='The hidden dimension of a neuron')
    parser.add_argument('--batch_size',type=int, default=500,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--decay_factor',type=float, default=0.95,
        metavar='--<decay factor>')
    parser.add_argument('log_dir', 
        metavar='<log directory>')
    parser.add_argument('model_dir', 
        metavar='<model directory>')
    parser.add_argument('feat_scp', 
        metavar='<feature scp file>')    
    return parser

def main():
    fn_list = \
    audio2vec.build_filename_list(FLAG.file_scp)
    #BN_evaluation(fn_list)
    return 

if __name__ == '__main__':
    parser = parser_opt()
    FLAG = parser.parse_args()
    main()

