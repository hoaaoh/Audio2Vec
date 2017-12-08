import tensorflow as tf
import numpy as np
import time
import os
import random
from datetime import datetime 
from model import Audio2Vec
from utils import *

n_files = 100
proportion = 0.9

class Solver(object):
    def __init__(self, feat_dir, train_feat_scp, test_feat_scp, batch_size, seq_len, feat_dim,
                 p_memory_dim, s_memory_dim, init_lr, log_dir, model_dir, n_epochs):
        self.feat_dir = feat_dir
        self.train_feat_scp = train_feat_scp
        self.test_feat_scp = test_feat_scp
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.p_memory_dim = p_memory_dim
        self.s_memory_dim = s_memory_dim
        self.init_lr = init_lr
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.n_epochs = n_epochs
        self.model = Audio2Vec(batch_size, p_memory_dim, s_memory_dim, seq_len, feat_dim)

        self.n_feats_train = None
        self.feats_train = None
        self.spk2feat_train = None
        self.feat2label_train = None
        self.spk_train = None
        self.n_batches_train = None
        self.n_feats_test = None
        self.feats_test = None
        self.spk2feat_test = None
        self.feat2label_test = None
        self.spk_test = None
        self.n_batches_test = None

    def reconstruct_opt(self, loss, learning_rate, momentum):
        ### Optimizer building              ###
        ### variable: train_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss)
        return train_op

    def generate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: generate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss, var_list=var_list)
        return train_op

    def discriminate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: discriminate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss)
        return train_op

    def BN_evaluation(self, saver, word_dir, utter_dir, reconstruction_loss, speaker_loss_pos, speaker_loss_neg,
                    generation_loss, discrimination_loss, summary_writer, summary_op,
                    p_enc, s_enc, save=False):
        """Getting Bottleneck Features"""
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            global_epoch = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print ("Model restored.")
            else:
                print ('No checkpoint file found.')
                return

            # Start evaluation.
            r_total_loss_value = 0.
            s_pos_total_loss_value = 0.
            s_neg_total_loss_value = 0.
            g_total_loss_value = 0.
            d_total_loss_value = 0.
            
            print ("Start loops in batches of test data!")
            for step in range(self.n_batches_test):
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size
                feat_indices = list(range(start_idx, end_idx))
                words = []
                utters = []
                for i in range(start_idx, end_idx):
                    words.append(self.feat2label_test[i][0])
                    utters.append(self.feat2label_test[i][1])
                batch_examples, batch_examples_pos, batch_examples_neg = \
                    batch_pair_data(self.feats_test, self.spk2feat_test,
                                    self.feat2label_test, feat_indices, self.spk_test)
                batch_examples = batch_examples.reshape((self.batch_size, self.seq_len, self.feat_dim))
                batch_examples_pos = batch_examples_pos.reshape((self.batch_size, self.seq_len, self.feat_dim))
                batch_examples_neg = batch_examples_neg.reshape((self.batch_size, self.seq_len, self.feat_dim))
                if summary_op != None:
                    summary_str, r_loss, s_loss_pos, s_loss_neg, g_loss, d_loss, p_memories, s_memories = \
                        sess.run([summary_op, reconstruction_loss, speaker_loss_pos, speaker_loss_neg, \
                                  generation_loss, discrimination_loss, p_enc, s_enc],
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                if save==True:
                    for i, word in enumerate(words):
                        p_single_memory = p_memories[i]
                        s_single_memory = s_memories[i]
                        p_single_memory = p_single_memory.tolist()
                        s_single_memory = s_single_memory.tolist()
                        with open(word_dir+'/'+str(word), 'a') as word_file:
                            for j, p in enumerate(p_single_memory):
                                word_file.write(str(p))
                                if j != len(p_single_memory)-1:
                                    word_file.write(',')
                                else:
                                    word_file.write('\n')
                        with open(utter_dir+'/'+str(utters[i]), 'a') as utter_file:
                            for j, s in enumerate(s_single_memory):
                                utter_file.write(str(s))
                                if j != len(s_single_memory)-1:
                                    utter_file.write(',')
                                else:
                                    utter_file.write('\n')
                            
                r_total_loss_value += r_loss
                s_pos_total_loss_value += s_loss_pos
                s_neg_total_loss_value += s_loss_neg
                g_total_loss_value += g_loss
                d_total_loss_value += d_loss
            r_avg_loss = r_total_loss_value/self.n_batches_test
            s_pos_avg_loss = s_pos_total_loss_value/self.n_batches_test
            s_neg_avg_loss = s_neg_total_loss_value/self.n_batches_test
            g_avg_loss = g_total_loss_value/self.n_batches_test
            d_avg_loss = d_total_loss_value/self.n_batches_test
            print ('%s: average r_loss for eval = %.3f' % (datetime.now(), r_avg_loss))
            print ('%s: average s_pos_loss for eval = %.3f' % (datetime.now(), s_pos_avg_loss))
            print ('%s: average s_neg_loss for eval = %.3f' % (datetime.now(), s_neg_avg_loss))
            print ('%s: average g_loss for eval = %.3f' % (datetime.now(), g_avg_loss))
            print ('%s: average d_loss for eval = %.3f' % (datetime.now(), d_avg_loss))
            if summary_op != None:
                summary = tf.Summary()
                summary.ParseFromString(summary_str)
                summary.value.add(tag='r eval loss', simple_value=r_avg_loss)
                summary.value.add(tag='s_pos eval loss', simple_value=s_pos_avg_loss)
                summary.value.add(tag='s_neg eval loss', simple_value=s_neg_avg_loss)
                summary.value.add(tag='g eval loss', simple_value=g_avg_loss)
                summary.value.add(tag='d eval loss', simple_value=d_avg_loss)
                summary_writer.add_summary(summary, global_epoch)

    def train(self):
        """ Training seq2seq for AudioVec."""
        reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, speaker_loss_pos, speaker_loss_neg, p_enc, s_enc = \
            self.model.build_model()

        # Build a graph that grains the model with one batch of examples and
        # updates the model parameters
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if not 'adversarial' in var.name]
        d_vars = [var for var in t_vars if 'adversarial' in var.name]
        
        generate_op = self.generate_opt(reconstruction_loss + generation_loss + speaker_loss_pos
                                   + speaker_loss_neg, self.init_lr, 0.9, g_vars)
        discriminate_op = self.discriminate_opt(discrimination_loss + 10*GP_loss,
                                           self.init_lr, 0.9, d_vars)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
        tf.summary.scalar("reconstruct loss", reconstruction_loss)
        tf.summary.scalar("speaker loss pos", speaker_loss_pos)
        tf.summary.scalar("speaker loss neg", speaker_loss_neg)
        tf.summary.scalar("generate loss", generation_loss)
        tf.summary.scalar("discriminate loss", discrimination_loss)
        tf.summary.scalar("GP loss", GP_loss)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        sess.graph.finalize()
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        ### Restore the model ###
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')

        ### Load data  ###
        feats_dir = os.path.join(self.feat_dir, 'feats', str(self.seq_len))
        split_data(self.feat_dir, n_files, proportion, self.seq_len)
        self.n_feats_train, self.feats_train, self.spk2feat_train, self.feat2label_train, self.spk_train \
            = load_data(feats_dir, self.train_feat_scp)
        self.n_feats_test, self.feats_test, self.spk2feat_test, self.feat2label_test, self.spk_test \
            = load_data(feats_dir, self.test_feat_scp)
        feat_order = list(range(self.n_feats_train))
        self.n_batches_train = self.n_feats_train // self.batch_size
        self.n_batches_test = self.n_feats_test // self.batch_size

        ### Start training ###
        print ("Start batch training.")
        for epoch in range(self.n_epochs):
            print ("Start of Epoch: " + str(epoch) + "!")
            random.shuffle(feat_order)
            for step in range(self.n_batches_train):
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size
                feat_indices = feat_order[start_idx:end_idx]
                batch_examples, batch_examples_pos, batch_examples_neg = \
                    batch_pair_data(self.feats_train, self.spk2feat_train,
                                    self.feat2label_train, feat_indices, self.spk_train)
                batch_examples = batch_examples.reshape((self.batch_size, self.seq_len, self.feat_dim))
                batch_examples_pos = batch_examples_pos.reshape((self.batch_size, self.seq_len, self.feat_dim))
                batch_examples_neg = batch_examples_neg.reshape((self.batch_size, self.seq_len, self.feat_dim))

                try:
                    start_time = time.time()
                    _, r_loss, g_loss, s_pos_loss, s_neg_loss = \
                        sess.run([generate_op, reconstruction_loss, generation_loss, \
                                  speaker_loss_pos, speaker_loss_neg], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                    for ite in range(5):
                        _, d_loss, gp_loss = sess.run([discriminate_op, discrimination_loss, GP_loss],
                                                      feed_dict={self.model.feat: batch_examples,
                                                                 self.model.feat_pos: batch_examples_pos,
                                                                 self.model.feat_neg: batch_examples_neg})
                    if step % 100 == 0:
                        duration = time.time() - start_time
                        example_per_sec = self.batch_size / duration
                        format_str = ('%s:epoch %d,step %d,\nr_loss=%.5f,s_pos_loss=%.5f,'
                                      's_neg_loss=%.5f,g_loss=%.5f,d_loss=%.5f,gp_loss=%.5f')
                        print (format_str % (datetime.now(), epoch, step, \
                                             r_loss, s_pos_loss, s_neg_loss, g_loss, d_loss, gp_loss))

                except tf.errors.OutOfRangeError:
                    break

            summary_str = sess.run(summary_op,feed_dict={self.model.feat: batch_examples,
                                                         self.model.feat_pos: batch_examples_pos,
                                                         self.model.feat_neg: batch_examples_neg})
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()
            ckpt = self.model_dir + '/model.ckpt'
            saver.save(sess, ckpt, global_step=step)
            summary_writer_eval = tf.summary.FileWriter(self.log_dir + '/BN', sess.graph)
            print ("Start evaluation!")
            self.BN_evaluation(saver, None, None, reconstruction_loss, speaker_loss_pos, 
                               speaker_loss_neg, generation_loss, discrimination_loss, 
                               summary_writer_eval, summary_op, p_enc, s_enc)
            print ("End of Epoch: " + str(epoch) + "!")
        summary_writer.flush()

    def test(self, word_dir, utter_dir):
        """ Testing seq2seq for AudioVec."""
        reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, speaker_loss_pos, speaker_loss_neg, p_enc, s_enc = \
            self.model.build_model()

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        ### Restore the model ###
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')

        ### Load data  ###
        feats_dir = os.path.join(self.feat_dir, 'feats', str(self.seq_len))
        self.n_feats_test, self.feats_test, self.spk2feat_test, self.feat2label_test, self.spk_test \
            = load_data(feats_dir, self.test_feat_scp)
        self.n_batches_test = self.n_feats_test // self.batch_size

        print ("Start evaluation!")
        self.BN_evaluation(saver, word_dir, utter_dir, reconstruction_loss, speaker_loss_pos, 
                           speaker_loss_neg, generation_loss, discrimination_loss, 
                           None, None, p_enc, s_enc)
        print ("End of evaluation!")
