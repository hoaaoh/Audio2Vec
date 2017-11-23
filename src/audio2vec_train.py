#!/usr/bin/env python3
import tensorflow as tf
import tempfile
import numpy as np
import time
import os
from math import floor
from datetime import datetime 
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import dtypes 
from tensorflow.python.ops import variable_scope

# from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

# import seq2seq 
from tensorflow.python.client import timeline
from tensorflow.python.ops import math_ops
from six.moves import xrange
import argparse

from flip_gradient import flip_gradient


log_file = None
model_file = None

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=2640000
NUM_EPOCHS_PER_DECAY= 1000.0
INITIAL_LEARNING_RATE= 0.1
LEARNING_RATE_DECAY_FACTOR = 0.95
MAX_STEP=70000
# NUM_UP_TO=100
FLAG = None

### data parsing ####
def TFRQ_feeding(filename_queue, feat_dim, seq_len):
    """ Reads and parse the examples from alignment dataset 
        in TF record format 
    Args: 
      filename_queue: A queue of strings with the filenames to read from 
      feat_dim : feature dimension 
      seq_len : sequence length (padded)

    Returns:
      An object representing a single example
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    example = tf.parse_single_example(serialized_example,
        features={
            'feat': tf.FixedLenFeature([seq_len*3, feat_dim],dtype=tf.float32),
            'label':tf.FixedLenFeature([3],dtype=tf.int64),
            'utterance':tf.FixedLenFeature([3],dtype=tf.string)
        })
    return example['feat'], example['label'], example['utterance']

def batch_pipeline(filenames, batch_size, feat_dim, seq_len, \
    num_epochs=None):
    """ 
    Building Input data graph
    examples: list of 2-D tensors in batch
    labels: list of 2-D tensors in batch
    """
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    # results = tf.unstack(dataQ_feeding(filename_queue, feat_dim, seq_len))
    # result = dataQ_feeding(filename_queue, feat_dim, seq_len)
    result = TFRQ_feeding(filename_queue, feat_dim, seq_len)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #  from -- bigger means better shuffling but slower start up and
    # more memory used
    # capacity must be larger than min_after_dequeue and the amount larger
    # determines the maximum we will prefetch. Recommendation:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 8 * batch_size
    example_batch, label_batch, utterance_batch = tf.train.shuffle_batch(
        result, batch_size=batch_size,num_threads=6,
                capacity=capacity,\
        min_after_dequeue=min_after_dequeue)
    example_batch = tf.transpose (example_batch, perm=[1,0,2])
    label_batch = tf.transpose (label_batch, perm=[1,0])
    utterance_batch = tf.transpose (utterance_batch, perm=[1,0])
    
    
    ### do batch normalization ###
    
    
    ### done batch normalization ###

    unstacked_examples = tf.unstack(example_batch, seq_len*3)
    unstacked_labels = tf.unstack(label_batch, 3)
    unstacked_utterances = tf.unstack(utterance_batch, 3)
    ### labels do not need to be unstacked ###
    ### unstacked_labels   = tf.unstack(label_batch, seq_len) ###
    return unstacked_examples, unstacked_labels, unstacked_utterances

def build_filename_list(list_fn):
    fn_list = []
    with open(list_fn,'r') as f:
        for line in f:
            fn_list.append(line.rstrip())
    return  fn_list

def loss(dec_out, labels, seq_len, batch_size, feat_dim):
    """ Build loss graph
    Args: 
      dec_out: decoder output sequences, list of 2-D tensor 
      labels : true label sequence, list of 2-D tensor 
    Return:
      loss 
    """
    labels_trans = tf.transpose(tf.reshape(labels, shape=(seq_len*batch_size, feat_dim)))
    labels_trans = tf.reshape(labels_trans, shape=[-1])
    dec_proj_outputs = tf.reshape(dec_out, shape=[-1])

    ### compute RMSE error ###
    ### mask the zeroes while computing loss ###
    zero = tf.constant(0.,dtype=tf.float32)
    where_no_mask = tf.cast(tf.not_equal(labels_trans,zero),dtype=tf.float32)
    dec_proj_outputs_masked = tf.multiply(where_no_mask, dec_proj_outputs)
    nums = tf.reduce_sum(where_no_mask)
    tmp_loss = tf.subtract(dec_proj_outputs_masked, labels_trans)
    tmp_loss = tf.multiply(tmp_loss, tmp_loss)

    loss = tf.sqrt(tf.divide(tf.reduce_sum(tmp_loss),nums), name='total_loss')
    return loss

def reconstruct_opt(loss, learning_rate, momentum):
    ### Optimizer building              ###
    ### variable: train_op              ###
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # train_op = optimizer.minimize(loss)
    return train_op

def generate_opt(loss, learning_rate, momentum, var_list):
    ### Optimizer building              ###
    ### variable: generate_op              ###
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # train_op = optimizer.minimize(loss, var_list=var_list)
    return train_op

def discriminate_opt(loss, learning_rate, momentum, var_list):
    ### Optimizer building              ###
    ### variable: discriminate_op              ###
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # train_op = optimizer.minimize(loss)
    return train_op

def encode(cell, examples):
    # examples_norm = tf.contrib.layers.layer_norm(examples)
    _, enc_state = core_rnn.static_rnn(cell, examples, dtype=dtypes.float32)
    # _, (c, enc_state) = core_rnn.static_rnn(cell, examples, dtype=dtypes.float32)
    return enc_state

def decode(cell, examples, batch_size, memory_dim, seq_len, feat_dim, enc_memory):
    dec_inp = (tf.unstack(tf.zeros_like(examples[:], dtype=tf.float32, name="GO")))
    dec_outputs, dec_state = seq2seq.rnn_decoder(dec_inp, enc_memory, cell)
    print (dec_state.shape)
    dec_reshape = tf.transpose(tf.reshape(dec_outputs, (seq_len*batch_size, memory_dim)))
    W_p = tf.get_variable("output_proj_w", [feat_dim, memory_dim])
    b_p = tf.get_variable("output_proj_b", shape=(feat_dim), initializer=tf.constant_initializer(0.0))
    b_p = [ b_p for i in range(seq_len*batch_size)]
    b_p = tf.transpose(b_p)
    dec_proj_outputs = tf.matmul(W_p, dec_reshape) + b_p
    return dec_proj_outputs

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

def train(fn_list, batch_size, memory_dim, seq_len=50, feat_dim=39, split_enc=20, gradient_flip=1.0):
    """ Training seq2seq for number of steps."""
    with tf.Graph().as_default():
        # global_step = tf.Variable(0, trainable=False)
        # get examples and labels for seq2seq #
        ########
        #/TODO #
        ########
        examples, labels, utterances = batch_pipeline(fn_list, batch_size, feat_dim, seq_len)
        examples_pos = [examples[i] for i in range(seq_len*3) if i%3 == 1]
        examples_neg = [examples[i] for i in range(seq_len*3) if i%3 == 2]
        examples = [examples[i] for i in range(seq_len*3) if i%3 == 0]
        labels_pos = labels[1]
        labels_neg = labels[2]
        labels = labels[0]
        utterances_pos = utterances[1]
        utterances_neg = utterances[2]
        utterances = utterances[0]

        # dec_out, enc_memory = inference(examples, batch_size, memory_dim, seq_len, feat_dim)
        # build a graph that computes the results
        with tf.variable_scope('encoding') as scope_1:
            W_enc_p = tf.get_variable("enc_w_p", [memory_dim - split_enc, memory_dim - split_enc])
            b_enc_p = tf.get_variable("enc_b_p", shape=[memory_dim - split_enc])
            W_enc_s = tf.get_variable("enc_w_s", [split_enc, split_enc])
            b_enc_s = tf.get_variable("enc_b_s", shape=[split_enc])
            # training example
            # dec_out, enc_memory = inference(examples, batch_size, memory_dim, seq_len, feat_dim)
            # enc_memory = tf.layers.batch_normalization(encode(examples, memory_dim))
            with tf.variable_scope('encoding_p') as scope_1_1:
                cell = core_rnn_cell.GRUCell(memory_dim - split_enc, activation=tf.nn.relu)
                # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
                p_enc = encode(cell, examples)
                p_enc = leaky_relu(tf.matmul(p_enc, W_enc_p) + b_enc_p)
                scope_1_1.reuse_variables()
                p_enc_pos = encode(cell, examples_pos)
                p_enc_pos = leaky_relu(tf.matmul(p_enc_pos, W_enc_p) + b_enc_p)
                p_enc_neg = encode(cell, examples_neg)
                p_enc_neg = leaky_relu(tf.matmul(p_enc_neg, W_enc_p) + b_enc_p)
            with tf.variable_scope('encoding_s') as scope_1_2:
                cell = core_rnn_cell.GRUCell(split_enc, activation=tf.nn.relu)
                # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
                s_enc = encode(cell, examples)
                s_enc = leaky_relu(tf.matmul(s_enc, W_enc_s) + b_enc_s)
                scope_1_2.reuse_variables()
                s_enc_pos = encode(cell, examples_pos)
                s_enc_pos = leaky_relu(tf.matmul(s_enc_pos, W_enc_s) + b_enc_s)
                s_enc_neg = encode(cell, examples_neg)
                s_enc_neg = leaky_relu(tf.matmul(s_enc_neg, W_enc_s) + b_enc_s)

        # speaker_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.nn.sigmoid(s_enc), logits=s_enc_pos))
        # Hinge loss
        speaker_loss_pos = tf.losses.mean_squared_error(s_enc, s_enc_pos)
        speaker_loss_neg = - tf.minimum(tf.losses.mean_squared_error(s_enc, s_enc_neg), tf.constant(0.01))
        speaker_loss = speaker_loss_pos + speaker_loss_neg

        # domain-adversarial
        with tf.variable_scope('adversarial_phonetic') as scope_2:
            W_adv_1 = tf.get_variable("adv_w_1", [2*(memory_dim - split_enc), 256])
            b_adv_1 = tf.get_variable("adv_b_1", shape=[256])
            W_adv_2 = tf.get_variable("adv_w_2", [256, 256])
            b_adv_2 = tf.get_variable("adv_b_2", shape=[256])
            W_adv_3 = tf.get_variable("adv_w_3", [256, 256])
            b_adv_3 = tf.get_variable("adv_b_3", shape=[256])
            W_bin = tf.get_variable("bin_w", [256, 1])
            b_bin = tf.get_variable("bin_b", shape=[1])

            # WGAN gradient penalty
            with tf.variable_scope('gradient_penalty') as scope_2_1:
                alpha = tf.random_uniform(shape=[batch_size, 2*(memory_dim - split_enc)], minval=0., maxval=1.)
                pair_pos_stop = tf.stop_gradient(tf.concat([p_enc, p_enc_pos], 1))
                pair_neg_stop = tf.stop_gradient(tf.concat([p_enc, p_enc_neg], 1))
                pair_hat = alpha * pair_pos_stop + (1 - alpha) * pair_neg_stop
                # pair_hat_norm = tf.contrib.layers.layer_norm(pair_hat)
                pair_hat_l1 = leaky_relu(tf.matmul(pair_hat, W_adv_1) + b_adv_1)
                pair_hat_l2 = leaky_relu(tf.matmul(pair_hat_l1, W_adv_2) + b_adv_2)
                pair_hat_l3 = leaky_relu(tf.matmul(pair_hat_l2, W_adv_3) + b_adv_3)
                bin_hat = leaky_relu(tf.matmul(pair_hat_l3, W_bin) + b_bin)

            GP_loss = tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.gradients(bin_hat, pair_hat)[0]**2, axis=1)) - 1.)**2)

            # discriminator
            with tf.variable_scope('discriminator') as scope_2_2:
                pair_pos = flip_gradient(tf.concat([p_enc, p_enc_pos], 1), l=0.)
                # pair_pos_norm = tf.contrib.layers.layer_norm(pair_pos)
                pair_pos_l1 = leaky_relu(tf.matmul(pair_pos, W_adv_1) + b_adv_1)
                pair_pos_l2 = leaky_relu(tf.matmul(pair_pos_l1, W_adv_2) + b_adv_2)
                pair_pos_l3 = leaky_relu(tf.matmul(pair_pos_l2, W_adv_3) + b_adv_3)
                bin_pos = leaky_relu(tf.matmul(pair_pos_l3, W_bin) + b_bin)

                pair_neg = flip_gradient(tf.concat([p_enc, p_enc_neg], 1), l=0.)
                # pair_neg_norm = tf.contrib.layers.layer_norm(pair_neg)
                pair_neg_l1 = leaky_relu(tf.matmul(pair_neg, W_adv_1) + b_adv_1)
                pair_neg_l2 = leaky_relu(tf.matmul(pair_neg_l1, W_adv_2) + b_adv_2)
                pair_neg_l3 = leaky_relu(tf.matmul(pair_neg_l2, W_adv_3) + b_adv_3)
                bin_neg = leaky_relu(tf.matmul(pair_neg_l3, W_bin) + b_bin)

            # generate_loss = tf.losses.mean_squared_error(bin_pos, bin_neg)
            # discriminate_loss = - tf.losses.mean_squared_error(bin_pos, bin_neg)#  + 10 * GP_loss
            generate_loss = tf.reduce_mean(bin_pos - bin_neg)
            discriminate_loss = tf.reduce_mean(bin_neg - bin_pos)

            # phonetic_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(bin_pos), logits=bin_pos \
            #               + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(bin_pos), logits=bin_neg)
            # phonetic_loss = tf.divide(tf.reduce_sum(phonetic_loss), batch_size)
        # calculate loss
        # dec_out, enc_memory = inference(examples, batch_size, memory_dim, seq_len, feat_dim)
        W_dec = tf.get_variable("dec_w", [memory_dim, memory_dim])
        b_dec = tf.get_variable("dec_b", shape=[memory_dim])
        # dec_out = decode(examples, batch_size, memory_dim*2, seq_len, feat_dim, enc_memory)
        dec_state = leaky_relu(tf.matmul(tf.concat([s_enc,p_enc], 1), W_dec) + b_dec)
        cell = core_rnn_cell.GRUCell(memory_dim, activation=tf.nn.relu)
        # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(memory_dim, activation=tf.nn.relu)
        dec_out = decode(cell, examples, batch_size, memory_dim, seq_len, feat_dim, dec_state)
        reconstruct_loss = loss(dec_out, examples, seq_len, batch_size, feat_dim) 
        # total_loss = reconstruct_loss + speaker_loss
        ########
        # TODO/#
        ########

        ### learning rate decay ###
        learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("learning rate", learning_rate)

        # build a graph that grains the model with one batch of examples and
        # updates the model parameters
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if not 'adversarial' in var.name]
        d_vars = [var for var in t_vars if 'adversarial' in var.name]
        print ('g_vars:')
        for var in g_vars:
            print (var.name)
        print ('d_vars:')
        for var in d_vars:
            print (var.name)
        
        # reconstruct_op = reconstruct_opt(reconstruct_loss, learning_rate, 0.9)
        generate_op = generate_opt(reconstruct_loss+generate_loss+speaker_loss, learning_rate, 0.9, g_vars)
        discriminate_op = discriminate_opt(discriminate_loss+10*GP_loss, learning_rate, 0.9, d_vars)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        tf.summary.scalar("reconstruct loss", reconstruct_loss)
        tf.summary.scalar("speaker loss", speaker_loss)
        tf.summary.scalar("generate loss", generate_loss)
        tf.summary.scalar("discriminate loss", discriminate_loss)
        tf.summary.scalar("GP loss", GP_loss)
        # tf.summary.scalar("kl_divergence loss", kl_divergence_loss)
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

        ### restore the model ###
        if not os.path.exists(model_file):
            os.makedirs(model_file)
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
        feed_lr = INITIAL_LEARNING_RATE#*pow(LEARNING_RATE_DECAY_FACTOR,int(floor(global_step/NUM_EPOCHS_PER_DECAY)))

        ### start training ###
        for step in range(global_step, MAX_STEP):
            try:
                
                start_time = time.time()
                # _, r_loss = sess.run([reconstruct_op, reconstruct_loss], \
                                             # feed_dict={learning_rate: feed_lr})
                _, r_loss, g_loss, s_loss, s_pos_loss, s_neg_loss = \
                    sess.run([generate_op, reconstruct_loss, generate_loss, \
                              speaker_loss, speaker_loss_pos, speaker_loss_neg], \
                              feed_dict={learning_rate: feed_lr})
                for ite in range(5):
                    _, d_loss, gp_loss = sess.run([discriminate_op, discriminate_loss, GP_loss], \
                                                  feed_dict={learning_rate: feed_lr})
                
                duration = time.time() - start_time
                example_per_sec = batch_size / duration
                epoch = floor(batch_size * step / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

                format_str = ('%s:epoch %d,step %d,LR %.5f,r_loss=%.4f,s_loss=%.4f,'
                              's_pos_loss=%.4f,s_neg_loss=%.4f,g_loss=%.4f,d_loss=%.4f,gp_loss=%.4f')
                print (format_str % (datetime.now(), epoch, step, feed_lr, \
                                     r_loss, s_loss, s_pos_loss, s_neg_loss, g_loss, d_loss, gp_loss), end='\n')
                # create time line #
                #num_examples_per_step = batch_size
                #tl = timeline.Timeline(run_metadata.step_stats)
                #ctf = tl.generate_chrome_trace_format(show_memory=True)
                if step % 100 == 0:
                    summary_str = sess.run(summary_op,feed_dict={learning_rate:
                        feed_lr})
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()
                    if step % 1000 == 0:
                        ckpt = model_file + '/model.ckpt'
                        saver.save(sess, ckpt, global_step=step)
                    #with open('timeline_'+str(step)+'.json','w') as f:
                    #    f.write(ctf)
                '''
                if step % NUM_EPOCHS_PER_DECAY == NUM_EPOCHS_PER_DECAY -1 :
                    feed_lr *= LEARNING_RATE_DECAY_FACTOR
                '''
            except tf.errors.OutOfRangeError:
                break
        coord.request_stop()
        coord.join(threads)
        summary_writer.flush()
    return

def test_feed(fn_list, batch_size, memory_dim, seq_len=50, feat_dim=39, split_enc=50, gradient_flip=1.0):
    """ Training seq2seq for number of steps."""
    with tf.Graph().as_default():
        # global_step = tf.Variable(0, trainable=False)
        # get examples and labels for seq2seq #
        examples, labels = batch_pipeline(fn_list, batch_size, feat_dim, seq_len)
        examples_pos = [examples[i] for i in range(seq_len*3) if i%3 == 1]
        examples_neg = [examples[i] for i in range(seq_len*3) if i%3 == 2]
        examples = [examples[i] for i in range(seq_len*3) if i%3 == 0]
        labels_pos = labels[1]
        labels_neg = labels[2]
        labels = labels[0]

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

        ### restore the model ###
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
        feed_lr = INITIAL_LEARNING_RATE
        ### start training ###
        for step in range(global_step, MAX_STEP):
            try:
                
                start_time = time.time()
                _ = sess.run([examples,labels])
                
                duration = time.time() - start_time
                example_per_sec = batch_size / duration
                epoch = ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)
                format_str = ('%s: epoch %d, LR:%.7f, step %d, ( %.1f examples/sec;'
                    ' %.3f sec/batch)')
                
                print (format_str % (datetime.now(), epoch, feed_lr, step,
                    example_per_sec, float(duration)))
                
                
            except tf.errors.OutOfRangeError:
                break
        coord.request_stop()
        coord.join(threads)
        summary_writer.flush()
    return


def addParser():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Training Script')
    parser.add_argument('--init_lr',  type=float, default=0.1,
        metavar='<--initial learning rate>')
    parser.add_argument('--decay_rate',type=int, default=1000,
        metavar='learning rate decay per batch epoch') 
    parser.add_argument('--hidden_dim',type=int, default=100,
        metavar='<--hidden dimension>',
        help='The hidden dimension of a neuron')
    parser.add_argument('--batch_size',type=int, default=256,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--max_step',type=int, default=80000,
        metavar='--<max step for training>',
        help='The max step for training')
    parser.add_argument('--split_enc', type=int, default=20,
        metavar='splitting size of the encoded vector')
    parser.add_argument('--gradient_flip', type=float, default=0.1,
        metavar='gradient flipping of adversarial training')

    parser.add_argument('log_dir', 
        metavar='<log directory>')
    parser.add_argument('model_dir', 
        metavar='<model directory>')
    parser.add_argument('feat_scp', 
        metavar='<feature scp file>')    
    return parser

def main():

    train_fn_scp =  FLAG.feat_scp
    print (train_fn_scp)
    fn_list = build_filename_list(train_fn_scp)
    train(fn_list, FLAG.batch_size, FLAG.hidden_dim, split_enc=FLAG.split_enc, gradient_flip=FLAG.gradient_flip)
    with open(model_file+'/feat_dim','w') as f:
        f.write(str(FLAG.hidden_dim))
    with open(model_file+'/batch_size','w') as f:
        f.write(str(FLAG.batch_size))

    return 

if __name__ == '__main__':
    parser = addParser()
    FLAG = parser.parse_args()
    INITIAL_LEARNING_RATE= FLAG.init_lr
    NUM_EPOCHS_PER_DECAY = FLAG.decay_rate
    log_file = FLAG.log_dir
    model_file = FLAG.model_dir
    MAX_STEP=FLAG.max_step    
    main()


