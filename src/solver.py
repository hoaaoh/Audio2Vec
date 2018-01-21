import tensorflow as tf
import numpy as np
import time
import os
import random
from datetime import datetime 
from model import Audio2Vec
from utils import *
from collections import deque

n_files = 200
proportion = 0.975
gram_num = 2

class Solver(object):
    def __init__(self, model_type, stack_num, feat_dir, train_feat_scp, test_feat_scp, batch_size, seq_len, feat_dim,
                 p_memory_dim, s_memory_dim, init_lr, log_dir, model_dir, n_epochs):
        self.model_type = model_type
        self.stack_num=stack_num
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
        self.model = Audio2Vec(model_type, stack_num, p_memory_dim, s_memory_dim, seq_len, feat_dim, batch_size)

        self.generate_op = None
        self.discriminate_op = None

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
        
    def generate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: generate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=momentum)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss, var_list=var_list)
        return train_op

    def discriminate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: discriminate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=momentum)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss)
        return train_op

    def save_batch_BN(self, word_word_dir, word_spk_dir, spk_word_dir, spk_spk_dir, phonetic_file,
                      p_memories, s_memories, feat_indices):
        """Getting Bottleneck Features"""
        for i in range(len(feat_indices)):
            word = self.feat2label_test[feat_indices[i]][0]
            spk = self.feat2label_test[feat_indices[i]][1][:-4]
            p_single_memory = p_memories[i]
            s_single_memory = s_memories[i]
            p_single_memory = p_single_memory.tolist()
            s_single_memory = s_single_memory.tolist()
            with open(phonetic_file, 'a') as ph_file:
                for j, p in enumerate(p_single_memory):
                    ph_file.write(str(p) + ' ')
                    if j == len(p_single_memory)-1:
                        ph_file.write(str(int(float(word))) + '\n')
            if word_word_dir != None:
                with open(word_word_dir+'/'+str(word), 'a') as word_file:
                    for j, p in enumerate(p_single_memory):
                        word_file.write(str(p))
                        if j != len(p_single_memory)-1:
                            word_file.write(',')
                        else:
                            word_file.write('\n')
                with open(word_spk_dir+'/'+str(spk), 'a') as word_file:
                    for j, p in enumerate(p_single_memory):
                        word_file.write(str(p))
                        if j != len(p_single_memory)-1:
                            word_file.write(',')
                        else:
                            word_file.write('\n')
                with open(spk_word_dir+'/'+str(word), 'a') as spk_file:
                    for j, s in enumerate(s_single_memory):
                        spk_file.write(str(s))
                        if j != len(s_single_memory)-1:
                            spk_file.write(',')
                        else:
                            spk_file.write('\n')
                with open(spk_spk_dir+'/'+str(spk), 'a') as spk_file:
                    for j, s in enumerate(s_single_memory):
                        spk_file.write(str(s))
                        if j != len(s_single_memory)-1:
                            spk_file.write(',')
                        else:
                            spk_file.write('\n')

    def compute_loss(self, mode, sess, summary_writer, summary_op, epoch, reconstruction_loss, generation_loss,
                     speaker_loss_pos, speaker_loss_neg, discrimination_loss, GP_loss, p_enc, s_enc, 
                     word_word_dir, word_spk_dir, spk_word_dir, spk_spk_dir, phonetic_file):
        if mode == 'train':
            feat_order = list(range(self.n_feats_train))
            random.shuffle(feat_order)
            n_batches = self.n_batches_train
            feats = self.feats_train
            spk2feat = self.spk2feat_train
            feat2label = self.feat2label_train
            spk_list = self.spk_train
        else:
            feat_order = list(range(self.n_feats_test))
            n_batches = self.n_batches_test
            feats = self.feats_test
            spk2feat = self.spk2feat_test
            feat2label = self.feat2label_test
            spk_list = self.spk_test
        r_total_loss_value = 0.
        s_pos_total_loss_value = 0.
        s_neg_total_loss_value = 0.
        g_total_loss_value = 0.
        d_total_loss_value = 0.
        gp_total_loss_value = 0.
        summary = None
        for step in range(n_batches):
            start_idx = step * self.batch_size
            end_idx = start_idx + self.batch_size
            feat_indices = feat_order[start_idx:end_idx]
            if step == n_batches:
                feat_indices = feat_order[step * self.batch_size:]
            batch_size = len(feat_indices)
            # print (feat_indices)
            batch_examples, batch_examples_pos, batch_examples_neg = \
                batch_pair_data(feats, spk2feat, feat2label, feat_indices, spk_list)
            batch_examples = batch_examples.reshape((batch_size, self.seq_len, self.feat_dim))
            batch_examples_pos = batch_examples_pos.reshape((batch_size, self.seq_len, self.feat_dim))
            batch_examples_neg = batch_examples_neg.reshape((batch_size, self.seq_len, self.feat_dim))

            if mode == 'train':
                start_time = time.time()
                if self.model_type == 'default':
                    _, summary, r_loss, g_loss, s_pos_loss, s_neg_loss = \
                        sess.run([self.generate_op, summary_op, reconstruction_loss, generation_loss, \
                                  speaker_loss_pos, speaker_loss_neg], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                    for ite in range(3):
                        _, summary, d_loss, gp_loss = \
                            sess.run([self.discriminate_op, summary_op, discrimination_loss, GP_loss],
                                                      feed_dict={self.model.feat: batch_examples,
                                                                 self.model.feat_pos: batch_examples_pos,
                                                                 self.model.feat_neg: batch_examples_neg})
                elif self.model_type == 'noGAN':
                    _, summary, r_loss, s_pos_loss, s_neg_loss = \
                        sess.run([self.generate_op, summary_op, reconstruction_loss, speaker_loss_pos, speaker_loss_neg], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                    g_loss = 0.0
                    d_loss = 0.0
                    gp_loss = 0.0
                else:
                    _, summary, r_loss, = \
                        sess.run([self.generate_op, summary_op, reconstruction_loss], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                    g_loss = 0.0
                    s_pos_loss = 0.0
                    s_neg_loss = 0.0
                    d_loss = 0.0
                    gp_loss = 0.0

                if step % 100 == 0:
                    duration = time.time() - start_time
                    example_per_sec = batch_size / duration
                    format_str = ('%s:epoch %d,step %d,\nr_loss=%.5f,s_pos_loss=%.5f,'
                                  's_neg_loss=%.5f,g_loss=%.5f,d_loss=%.5f,gp_loss=%.5f')
                    print (format_str % (datetime.now(), epoch, step, \
                                         r_loss, s_pos_loss, s_neg_loss, g_loss, d_loss, gp_loss))
            elif mode == 'test':
                if summary_writer == None:
                    r_loss, g_loss, s_pos_loss, s_neg_loss, d_loss, gp_loss, p_memories, s_memories = \
                        sess.run([reconstruction_loss, generation_loss, \
                                  speaker_loss_pos, speaker_loss_neg, discrimination_loss, GP_loss, p_enc, s_enc], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
                    self.save_batch_BN(word_word_dir, word_spk_dir, spk_word_dir, spk_spk_dir, phonetic_file,
                                       p_memories, s_memories, feat_indices)
                else:
                    summary, r_loss, g_loss, s_pos_loss, s_neg_loss, d_loss, gp_loss, p_memories, s_memories = \
                        sess.run([summary_op, reconstruction_loss, generation_loss, \
                                  speaker_loss_pos, speaker_loss_neg, discrimination_loss, GP_loss, p_enc, s_enc], \
                                 feed_dict={self.model.feat: batch_examples,
                                            self.model.feat_pos: batch_examples_pos,
                                            self.model.feat_neg: batch_examples_neg})
            else:
                r_loss, g_loss, s_pos_loss, s_neg_loss, d_loss, gp_loss, p_memories, s_memories = \
                    sess.run([reconstruction_loss, generation_loss, \
                              speaker_loss_pos, speaker_loss_neg, discrimination_loss, GP_loss, p_enc, s_enc], \
                             feed_dict={self.model.feat: batch_examples,
                                        self.model.feat_pos: batch_examples_pos,
                                        self.model.feat_neg: batch_examples_neg})
                self.save_batch_BN(None, None, None, None, phonetic_file, p_memories, s_memories, feat_indices)
            r_total_loss_value += r_loss
            s_pos_total_loss_value += s_pos_loss
            s_neg_total_loss_value += s_neg_loss
            g_total_loss_value += g_loss
            d_total_loss_value += d_loss
            gp_total_loss_value += gp_loss
        r_avg_loss = r_total_loss_value/n_batches
        s_pos_avg_loss = s_pos_total_loss_value/n_batches
        s_neg_avg_loss = s_neg_total_loss_value/n_batches
        g_avg_loss = g_total_loss_value/n_batches
        d_avg_loss = d_total_loss_value/n_batches
        gp_avg_loss = gp_total_loss_value/n_batches
        if summary_writer != None:
            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()
        if mode != 'train':
            print ('%s: average r_loss for eval = %.5f' % (datetime.now(), r_avg_loss))
            print ('%s: average s_pos_loss for eval = %.5f' % (datetime.now(), s_pos_avg_loss))
            print ('%s: average s_neg_loss for eval = %.5f' % (datetime.now(), s_neg_avg_loss))
            print ('%s: average g_loss for eval = %.5f' % (datetime.now(), g_avg_loss))
            print ('%s: average d_loss for eval = %.5f' % (datetime.now(), d_avg_loss))
            print ('%s: average gp_loss for eval = %.5f' % (datetime.now(), gp_avg_loss))
        # return r_avg_loss, s_pos_avg_loss, s_neg_avg_loss, g_avg_loss, d_avg_loss, gp_avg_loss

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
        # print ("### G_vars ###")
        # for g in g_vars:
            # print (g)
        # print ("### D_vars ###")
        # for d in d_vars:
            # print (d)
        
        if self.model_type == 'default':
            self.generate_op = self.generate_opt(reconstruction_loss + generation_loss + speaker_loss_pos
                                       + speaker_loss_neg, self.init_lr, 0.9, g_vars)
            self.discriminate_op = self.discriminate_opt(discrimination_loss + 10*GP_loss,
                                       self.init_lr, 0.9, d_vars)
        else:
            self.generate_op = self.generate_opt(reconstruction_loss, self.init_lr, 0.9, g_vars)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)
        # sess.graph.finalize()

        summary_train = [tf.summary.scalar("reconstruction loss", reconstruction_loss),
                        tf.summary.scalar("speaker loss pos", speaker_loss_pos),
                        tf.summary.scalar("speaker loss neg", speaker_loss_neg),
                        tf.summary.scalar("generation loss", generation_loss),
                        tf.summary.scalar("discrimination loss", discrimination_loss),
                        tf.summary.scalar("GP loss", GP_loss)]
        summary_test = [tf.summary.scalar("reconstrucion loss eval", reconstruction_loss),
                        tf.summary.scalar("speaker loss pos eval", speaker_loss_pos),
                        tf.summary.scalar("speaker loss neg eval", speaker_loss_neg),
                        tf.summary.scalar("generation loss eval", generation_loss),
                        tf.summary.scalar("discrimination loss eval", discrimination_loss),
                        tf.summary.scalar("GP loss eval", GP_loss)]
        summary_op_train = tf.summary.merge(summary_train)
        summary_op_test = tf.summary.merge(summary_test)
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
        # split_data(self.feat_dir, n_files, proportion, self.seq_len)
        self.n_feats_train, self.feats_train, self.spk2feat_train, self.feat2label_train, self.spk_train \
            = load_data(feats_dir, self.train_feat_scp)
        self.n_feats_test, self.feats_test, self.spk2feat_test, self.feat2label_test, self.spk_test \
            = load_data(feats_dir, self.test_feat_scp)
        self.n_batches_train = self.n_feats_train // self.batch_size
        self.n_batches_test = self.n_feats_test // self.batch_size

        ### Start training ###
        print ("Start batch training.")
        for epoch in range(self.n_epochs):
            print ("Start of Epoch: " + str(epoch) + "!")
            self.compute_loss('train', sess, summary_writer, summary_op_train, epoch, reconstruction_loss, 
                              generation_loss, speaker_loss_pos, speaker_loss_neg ,discrimination_loss, 
                              GP_loss, p_enc, s_enc, None, None, None, None, None)
            self.compute_loss('test', sess, summary_writer, summary_op_test, epoch, reconstruction_loss, 
                              generation_loss, speaker_loss_pos, speaker_loss_neg ,discrimination_loss, 
                              GP_loss, p_enc, s_enc, None, None, None, None, None)

            ckpt = self.model_dir + '/model.ckpt'
            saver.save(sess, ckpt, global_step=epoch+global_step)
            print ("End of Epoch: " + str(epoch) + "!")
        summary_writer.flush()

    def test(self, word_word_dir, word_spk_dir, spk_word_dir, spk_spk_dir, phonetic_file):
        """ Testing seq2seq for AudioVec."""
        reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, speaker_loss_pos, speaker_loss_neg, p_enc, s_enc = \
            self.model.build_model()

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)
        sess.graph.finalize()

        ### Restore the model ###
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')
            return

        ### Load data  ###
        feats_dir = os.path.join(self.feat_dir, 'feats', str(self.seq_len))
        self.n_feats_test, self.feats_test, self.spk2feat_test, self.feat2label_test, self.spk_test \
            = load_data(feats_dir, self.test_feat_scp)
        self.n_batches_test = self.n_feats_test // self.batch_size

        ### Start testing ###
        self.compute_loss('test', sess, None, None, None, reconstruction_loss, generation_loss, \
                 speaker_loss_pos, speaker_loss_neg ,discrimination_loss, GP_loss, p_enc, s_enc, \
                          word_word_dir, word_spk_dir, spk_word_dir, spk_spk_dir, phonetic_file)
    
    def make_phonetic(self, all_AE_dir, phonetic_dir):
        """ Making phonetic for AudioVec."""
        reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, speaker_loss_pos, speaker_loss_neg, p_enc, s_enc = \
            self.model.build_model()

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)
        sess.graph.finalize()

        ### Restore the model ###
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')
            return
        
        ### ###
        feats_dir = os.path.join(self.feat_dir, 'feats', str(self.seq_len))
        for num, all_AE_scp in enumerate(os.listdir(all_AE_dir)):
            all_AE_scp = os.path.join(all_AE_dir, all_AE_scp)
            phonetic_file = os.path.join(phonetic_dir, 'phonetic_all_'+str(num))
            print (all_AE_scp)
            print (phonetic_file)

            ### Load data  ###
            self.n_feats_test, self.feats_test, self.spk2feat_test, self.feat2label_test, self.spk_test \
                = load_data(feats_dir, all_AE_scp)
            self.n_batches_test = self.n_feats_test // self.batch_size

            ### Start testing ###
            self.compute_loss('phonetic', sess, None, None, None, reconstruction_loss, generation_loss, \
                     speaker_loss_pos, speaker_loss_neg ,discrimination_loss, GP_loss, p_enc, s_enc, \
                              None, None, None, None, phonetic_file)
            self.n_feats_test = None
            self.feats_test = None
            self.spk2feat_test = None
            self.feat2label_test = None
            self.spk_test = None
            self.n_batches_test = None


