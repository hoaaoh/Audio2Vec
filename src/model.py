import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import dtypes 
import copy
        
class Audio2Vec(object):
    def __init__(self, batch_size, p_memory_dim, s_memory_dim, seq_len, feat_dim):
        self.batch_size = batch_size
        self.p_memory_dim = p_memory_dim
        self.s_memory_dim = s_memory_dim
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        self.feat = tf.placeholder(tf.float32, [None, seq_len, feat_dim])
        self.feat_pos = tf.placeholder(tf.float32, [None, seq_len, feat_dim])
        self.feat_neg = tf.placeholder(tf.float32, [None, seq_len, feat_dim])

    def leaky_relu(self, x, alpha=0.01):
        return tf.maximum(x, alpha*x)

    def rnn_encode(self, cell, feat, stack_num=1):
        # examples_norm = tf.contrib.layers.layer_norm(examples)
        # _, (c, enc_state) = core_rnn.static_rnn(cell, examples, dtype=dtypes.float32)
        feat_perm = tf.transpose (feat, perm=[1,0,2])
        unstacked_feat = tf.unstack(feat_perm, self.seq_len)
        with tf.variable_scope("stack_rnn_encoder"):
            enc_cell = copy.copy(cell)
            enc_output, enc_state = core_rnn.static_rnn(enc_cell, unstacked_feat, dtype=dtypes.float32)
            for i in range(2, stack_num):
                with tf.variable_scope("stack_rnn_encoder_"+str(i)):
                    enc_cell = copy.copy(cell)
                    enc_output, enc_state = core_rnn.static_rnn(enc_cell, enc_output, dtype=dtypes.float32)
        return enc_state

    def encode(self, feat, feat_pos, feat_neg):
        with tf.variable_scope('encode') as scope_1:
            W_enc_p = tf.get_variable("enc_w_p", [self.p_memory_dim, self.p_memory_dim])
            b_enc_p = tf.get_variable("enc_b_p", shape=[self.p_memory_dim])
            W_enc_s = tf.get_variable("enc_w_s", [self.s_memory_dim, self.s_memory_dim])
            b_enc_s = tf.get_variable("enc_b_s", shape=[self.s_memory_dim])

            with tf.variable_scope('encode_p') as scope_1_1:
                cell = core_rnn_cell.GRUCell(self.p_memory_dim, activation=tf.nn.relu)
                # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
                p_enc = self.rnn_encode(cell, feat)
                p_enc = self.leaky_relu(tf.matmul(p_enc, W_enc_p) + b_enc_p)
                scope_1_1.reuse_variables()
                p_enc_pos = self.rnn_encode(cell, feat_pos)
                p_enc_pos = self.leaky_relu(tf.matmul(p_enc_pos, W_enc_p) + b_enc_p)
                p_enc_neg = self.rnn_encode(cell, feat_neg)
                p_enc_neg = self.leaky_relu(tf.matmul(p_enc_neg, W_enc_p) + b_enc_p)
            with tf.variable_scope('encode_s') as scope_1_2:
                cell = core_rnn_cell.GRUCell(self.s_memory_dim, activation=tf.nn.relu)
                # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
                s_enc = self.rnn_encode(cell, feat)
                s_enc = self.leaky_relu(tf.matmul(s_enc, W_enc_s) + b_enc_s)
                scope_1_2.reuse_variables()
                s_enc_pos = self.rnn_encode(cell, feat_pos)
                s_enc_pos = self.leaky_relu(tf.matmul(s_enc_pos, W_enc_s) + b_enc_s)
                s_enc_neg = self.rnn_encode(cell, feat_neg)
                s_enc_neg = self.leaky_relu(tf.matmul(s_enc_neg, W_enc_s) + b_enc_s)
        return p_enc, p_enc_pos, p_enc_neg, s_enc, s_enc_pos, s_enc_neg

    def gradient_penalty(self, W_adv_1, b_adv_1, W_adv_2, b_adv_2,
                         W_bin, b_bin, p_enc, p_enc_pos, p_enc_neg):
        with tf.variable_scope('gradient_penalty') as scope_2_1:
            alpha = tf.random_uniform(shape=[self.batch_size, 2*self.p_memory_dim], minval=0., maxval=1.)
            pair_pos_stop = tf.stop_gradient(tf.concat([p_enc, p_enc_pos], 1))
            pair_neg_stop = tf.stop_gradient(tf.concat([p_enc, p_enc_neg], 1))
            pair_hat = alpha * pair_pos_stop + (1 - alpha) * pair_neg_stop
            # pair_hat_norm = tf.contrib.layers.layer_norm(pair_hat)
            pair_hat_l1 = self.leaky_relu(tf.matmul(pair_hat, W_adv_1) + b_adv_1)
            pair_hat_l2 = self.leaky_relu(tf.matmul(pair_hat_l1, W_adv_2) + b_adv_2)
            bin_hat = self.leaky_relu(tf.matmul(pair_hat_l2, W_bin) + b_bin)

        GP_loss = tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.gradients(bin_hat, pair_hat)[0]**2, axis=1)) - 1.)**2)
        return GP_loss

    def discriminate(self, W_adv_1, b_adv_1, W_adv_2, b_adv_2,
                     W_bin, b_bin, p_enc, p_enc_pos, p_enc_neg):
        with tf.variable_scope('discriminate') as scope_2_2:
            pair_pos = tf.concat([p_enc, p_enc_pos], 1)
            # pair_pos_norm = tf.contrib.layers.layer_norm(pair_pos)
            pair_pos_l1 = self.leaky_relu(tf.matmul(pair_pos, W_adv_1) + b_adv_1)
            pair_pos_l2 = self.leaky_relu(tf.matmul(pair_pos_l1, W_adv_2) + b_adv_2)
            bin_pos = self.leaky_relu(tf.matmul(pair_pos_l2, W_bin) + b_bin)

            pair_neg = tf.concat([p_enc, p_enc_neg], 1)
            # pair_neg_norm = tf.contrib.layers.layer_norm(pair_neg)
            pair_neg_l1 = self.leaky_relu(tf.matmul(pair_neg, W_adv_1) + b_adv_1)
            pair_neg_l2 = self.leaky_relu(tf.matmul(pair_neg_l1, W_adv_2) + b_adv_2)
            bin_neg = self.leaky_relu(tf.matmul(pair_neg_l2, W_bin) + b_bin)

        # generate_loss = tf.losses.mean_squared_error(bin_pos, bin_neg)
        # discriminate_loss = - tf.losses.mean_squared_error(bin_pos, bin_neg)#  + 10 * GP_loss
        discrimination_loss = tf.reduce_mean(bin_pos - bin_neg)
        return discrimination_loss

    def adversarial_training(self, p_enc, p_enc_pos, p_enc_neg):
        with tf.variable_scope('adversarial_phonetic') as scope_2:
            W_adv_1 = tf.get_variable("adv_w_1", [2*(self.p_memory_dim), self.p_memory_dim/2])
            b_adv_1 = tf.get_variable("adv_b_1", shape=[self.p_memory_dim/2])
            W_adv_2 = tf.get_variable("adv_w_2", [self.p_memory_dim/2, self.p_memory_dim/2])
            b_adv_2 = tf.get_variable("adv_b_2", shape=[self.p_memory_dim/2])
            W_bin = tf.get_variable("bin_w", [self.p_memory_dim/2, 1])
            b_bin = tf.get_variable("bin_b", shape=[1])

            # WGAN gradient penalty
            GP_loss = self.gradient_penalty(W_adv_1, b_adv_1, W_adv_2, b_adv_2,
                         W_bin, b_bin, p_enc, p_enc_pos, p_enc_neg)
            # discrimination and generation loss
            discrimination_loss = self.discriminate(W_adv_1, b_adv_1, W_adv_2, b_adv_2,
                     W_bin, b_bin, p_enc, p_enc_pos, p_enc_neg)
        return GP_loss, discrimination_loss

    def rnn_decode(self, cell, feat, enc_memory, stack_num=1):
        dec_inp = (tf.unstack(tf.zeros([self.seq_len, self.batch_size, self.feat_dim], dtype=tf.float32, name="GO")))
        with tf.variable_scope("stack_rnn_decoder"):
            dec_cell = copy.copy(cell)
            dec_output, dec_state = seq2seq.rnn_decoder(dec_inp, enc_memory, dec_cell)
            for i in range(2, stack_num):
                with tf.variable_scope("stack_rnn_decoder_"+str(i)):
                    dec_cell = copy.copy(cell)
                    dec_output, dec_state = core_rnn.static_rnn(dec_cell, dec_output, dtype=dtypes.float32)
            dec_reshape = tf.transpose(tf.reshape(dec_output,
                (self.seq_len*self.batch_size, self.p_memory_dim+self.s_memory_dim)))
            W_p = tf.get_variable("output_proj_w",
                                  [self.feat_dim, self.p_memory_dim+self.s_memory_dim])
            b_p = tf.get_variable("output_proj_b", shape=(self.feat_dim),
                                  initializer=tf.constant_initializer(0.0))
            b_p = [ b_p for i in range(self.seq_len*self.batch_size)]
            b_p = tf.transpose(b_p)
            dec_proj_outputs = tf.matmul(W_p, dec_reshape) + b_p
        return dec_proj_outputs

    def decode(self, p_enc, s_enc, feat):
        with tf.variable_scope('decode') as scope_3:
            W_dec = tf.get_variable("dec_w", [self.p_memory_dim+self.s_memory_dim, \
                                              self.p_memory_dim+self.s_memory_dim])
            b_dec = tf.get_variable("dec_b", shape=[self.p_memory_dim+self.s_memory_dim])

            dec_state = self.leaky_relu(tf.matmul(tf.concat([p_enc,s_enc], 1), W_dec) + b_dec)
            cell = core_rnn_cell.GRUCell(self.p_memory_dim+self.s_memory_dim, activation=tf.nn.relu)
            # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
            dec_out = self.rnn_decode(cell, feat, dec_state)
        return dec_out

    def rec_loss(self, dec_out, feat):
        """ Build loss graph
        Args: 
          dec_out: decoder output sequences, list of 2-D tensor 
          labels : true label sequence, list of 2-D tensor 
        Return:
          loss 
        """

        labels_trans = tf.transpose(tf.reshape(feat, 
                shape=(self.seq_len*self.batch_size, self.feat_dim)))
        labels_trans = tf.reshape(labels_trans, shape=[-1])
        dec_proj_outputs = tf.reshape(dec_out, shape=[-1])
        ### compute RMSE error ###
        ### mask the zeroes while computing loss ###
        zero = tf.constant(0.,dtype=tf.float32)
        where_no_mask = tf.cast(tf.not_equal(labels_trans, zero),dtype=tf.float32)
        dec_proj_outputs_masked = tf.multiply(where_no_mask, dec_proj_outputs)
        nums = tf.reduce_sum(where_no_mask)
        tmp_loss = tf.subtract(dec_proj_outputs_masked, labels_trans)
        tmp_loss = tf.multiply(tmp_loss, tmp_loss)

        reconstruction_loss = tf.sqrt(tf.divide(tf.reduce_sum(tmp_loss),nums),
                                      name='reconstruction_loss')
        return reconstruction_loss

    def build_model(self):
        feat = self.feat
        feat_pos = self.feat_pos
        feat_neg = self.feat_neg

        # Encode
        p_enc, p_enc_pos, p_enc_neg, s_enc, s_enc_pos, s_enc_neg = \
            self.encode(feat, feat_pos, feat_neg)

        # Hinge loss
        speaker_loss_pos = tf.losses.mean_squared_error(s_enc, s_enc_pos)
        speaker_loss_neg = - tf.reduce_mean(tf.maximum(tf.constant(0.01)
                            - tf.norm(s_enc - s_enc_neg, axis=1), tf.constant(0.)))

        # Adversarial training
        GP_loss, discrimination_loss = self.adversarial_training(p_enc, p_enc_pos, p_enc_neg)
        generation_loss = - discrimination_loss 

        # Reconstruction loss
        dec_out = self.decode(p_enc, s_enc_pos, feat)
        reconstruction_loss = self.rec_loss(dec_out, feat)

        return reconstruction_loss, generation_loss, discrimination_loss, \
            GP_loss, speaker_loss_pos, speaker_loss_neg, p_enc, s_enc
