#!/usr/bin/env python3
import tensorflow as tf
import tempfile
import numpy as np
import time 
from math import floor
from datetime import datetime 
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

import seq2seq 
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
def dataQ_feeding(filename_queue, feat_dim, seq_len):
    """ Reads and parse the examples from alignment dataset 
    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        MFCC sequence: 200 * 39 dimensions 
        
    """
    class MFCCRECORD(object):
        pass 
    
    result = MFCCRECORD()

    ### use the line reader ### 
    reader = tf.TextLineReader()
    #values = []
    #for i in range(NUM_UP_TO):
    #    key, value = reader.read(filename_queue)
    #    values.append(value)
    key, value = reader.read(filename_queue)
    ### try to read NUM_UP_TO lines in one time ###
    ### read the csv file into features ###
    # seq = []
    record_defaults = [[1.] for i in range(feat_dim*seq_len)]
    # for value in values:
    #    seq.append(tf.decode_csv(value, record_defaults=record_defaults))
    tmp_result = tf.decode_csv(value, record_defaults=record_defaults) 
    ### so we have (NUM_UP_TO, seq_len *feat_dim ) ###
    ### reshape it into (NUM_UP_TO, seq_len, feat_dim) ###
    ### result.mfcc: sequence ###
    mfcc = tf.cast(tf.reshape(tmp_result, shape=(seq_len , \
        feat_dim)),tf.float32)
    ### result.rev_mfcc: reverse of sequence ###
    # result.rev_mfcc = tf.reverse(result.mfcc, [False, True])
    return mfcc, mfcc

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
            'label':tf.FixedLenFeature([3],dtype=tf.int64)
        })
    return example['feat'], example['label']

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
    example_batch, label_batch = tf.train.shuffle_batch(
        result, batch_size=batch_size,num_threads=6,
                capacity=capacity,\
        min_after_dequeue=min_after_dequeue)
    example_batch = tf.transpose (example_batch, perm=[1,0,2])
    label_batch = tf.transpose (label_batch, perm=[1,0])
    
    
    ### do batch normalization ###
    
    
    ### done batch normalization ###

    unstacked_examples = tf.unstack(example_batch, seq_len*3)
    unstacked_labels = tf.unstack(label_batch, 3)
    ### labels do not need to be unstacked ###
    ### unstacked_labels   = tf.unstack(label_batch, seq_len) ###
    return unstacked_examples, unstacked_labels


def inference_test(examples, batch_size, memory_dim, seq_len, feat_dim):
    """ test inference without transpose matrix 

    """
    dec_inp = ([tf.zeros_like(examples[0], dtype=tf.float32,
        name="GO")]+examples[:-1])
    
    cell = core_rnn_cell.GRUCell(memory_dim)
    dec_outputs, enc_memory,  dec_memory = seq2seq.basic_rnn_seq2seq_with_bottle_memory(examples, dec_inp, cell)
    ### no transition here ###

    return dec_outputs

def loss_tmp(dec_out, labels, seq_len, batch_size, feat_dim):
    """ Build loss graph
    Args: 
      dec_out: decoder output sequences, list of 2-D tensor labels : true label sequence, list of 2-D tensor 
    Return:
      loss 
    """
    labels_trans = tf.transpose(tf.reshape(labels, shape=(seq_len*batch_size, feat_dim)))
    labels_trans = tf.reshape(labels_trans, shape=[-1])
    dec_proj_outputs = tf.reshape(dec_out, shape=[-1])

    ### compute RMSE error ###
    ### mask the zeroes while computing loss ###
    #zero = tf.constant(0,dtype=tf.float32)
    #where_no_mask = tf.cast(tf.not_equal(labels_trans,zero),dtype=tf.float32)
    #dec_proj_outputs_masked = tf.multiply(where_no_mask, dec_proj_outputs)
    #nums = tf.reduce_sum(where_no_mask)
    tmp_loss = tf.subtract(dec_proj_outputs, labels_trans)
    tmp_loss = tf.multiply(tmp_loss, tmp_loss)
    nums = tf.constant(batch_size*seq_len*feat_dim, dtype=tf.float32)
    loss = tf.sqrt(tf.divide(tf.reduce_sum(tmp_loss),nums), name='total_loss')
    
    return loss


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

def memory_regularizer(enc_mem, batch_size, seq_len, feat_dim):
    ''' using the memory as the regularization term 
    Args:
      enc_mem: the encoder memories

    Return:
      reg: the regularization term 
    '''
    


    return 

def train_opt(loss, learning_rate, momentum):
    ### Optimizer building              ###
    ### variable: train_op              ###
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -100., 100.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    #train_op = optimizer.minimize(loss)

    return train_op

def loggin(graph):
    logdir = tempfile.mkdtemp()
    print (logdir)
    summary_writer = tf.train.SummaryWriter(logdir, graph)
    return summary_writer

def batch_training(batch_size, feat_dim, seq_len, enc_inp, labels, train_op,
    loss,  summary_op,sess):
    ### example training ###
    X = [[ np.random.choice(2, size=(feat_dim,), replace=True)
         for _ in range(batch_size)] for j in range(seq_len)]
    Y = X[:]
    # print (np.shape(X), np.shape(Y))
    feed_dict = { enc_inp[t]: X[t] for t in range(seq_len) }
    feed_dict.update({labels[t] : Y[t] for t in range(seq_len) })

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

def build_filename_list(list_fn):
    fn_list = []
    with open(list_fn,'r') as f:
        for line in f:
            fn_list.append(line.rstrip())

    return  fn_list


def add_loss_summary(total_loss):
    """Add summaries for losses in seq2seq model.

    Generates moving average for all losses and associated summaries for 
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_avarages_op: op fo generating moving averages of losses.
    """
    # Compute the moving average of all indiidual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')

    return 

def create_batch_file(fn_list, step, batch_size, seq_len, feat_dim, example):
    length = len(fn_list)
    print (fn_list[step%length])

    with open(fn_list[step%length]) as f:
        ln_cnt = 0
        for line in f:
            line_sp = line.rstrip().split(',')
            for i in xrange(len(line_sp)):
                example[ln_cnt][int(floor(float(i)/feat_dim))][i%feat_dim] = float(line_sp[i])
            ln_cnt += 1
    # print (example)
    return 

def create_train(total_loss, global_step,batch_size):
    """ 
    Create an optimizer and apply to all trainable variables.
    Add moving average for all trainable variables.

    Args: 
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training step
        processed.
    Returns:
      train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,
                                    decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries

    return 

def inference(examples, batch_size, memory_dim, seq_len, feat_dim):
    """ Build the seq2seq model 
    Args: 
      Sequence Inputs: list of 2-D tensors
      batch_size
      memory_dim
      feat_dim 

    Returns:
      Sequence Results: list of 2-D tensors
    """
    ### Decoder input: prepend all  "GO" tokens and drop the final    ###
    ### token of the encoder input                                    ###
    ### input: GO GO GO GO GO ... GO                                  ###
    dec_inp = (tf.unstack(tf.zeros_like(examples[:], dtype=tf.float32,
        name="GO")))
    #dec_inp = ([tf.zeros_like(examples[0], dtype=tf.float32,
    #    name="GO")] + examples[:-1])

    ### these two calls defined main cell in seq2seq and seq2seq model ###
    cell = core_rnn_cell.GRUCell(memory_dim, activation=tf.nn.relu)

    dec_outputs, enc_memory, dec_memory = seq2seq.basic_rnn_seq2seq_with_bottle_memory(examples, dec_inp, cell)
    ######################################################################
    
    dec_reshape = tf.transpose(tf.reshape(dec_outputs, (seq_len*batch_size,\
            memory_dim)))
    W_p = tf.get_variable("output_proj_w", [feat_dim, memory_dim])
    b_p = tf.get_variable("output_proj_b", shape=(feat_dim), \
            initializer=tf.constant_initializer(0.0))
    b_p = [ b_p for i in range(seq_len*batch_size)]
    b_p = tf.transpose(b_p)
    dec_proj_outputs = tf.matmul(W_p, dec_reshape) + b_p

    return dec_proj_outputs, enc_memory

def train(fn_list, batch_size, memory_dim, seq_len=50, feat_dim=39, split_enc=50):
    """ Training seq2seq for number of steps."""
    with tf.Graph().as_default():
        # global_step = tf.Variable(0, trainable=False)
        # get examples and labels for seq2seq #
        ########
        #/TODO #
        ########
        examples, labels = batch_pipeline(fn_list, batch_size, feat_dim, seq_len)
        examples_pos = [examples[i] for i in range(seq_len*3) if i%3 == 1]
        examples_neg = [examples[i] for i in range(seq_len*3) if i%3 == 2]
        examples = [examples[i] for i in range(seq_len*3) if i%3 == 0]
        labels_pos = labels[1]
        labels_neg = labels[2]
        labels = labels[0]

        # build a graph that computes the results
        with tf.variable_scope('reconstruction_and_speaker') as scope_1:
            # training example
            dec_out, enc_memory = inference(examples, batch_size, memory_dim, seq_len, feat_dim)
            print (enc_memory.get_shape())
            s_enc = tf.slice(enc_memory, [0, 0], [batch_size, split_enc])
            p_enc = tf.slice(enc_memory, [0, split_enc], [batch_size, memory_dim-split_enc])
            scope_1.reuse_variables()
            # positive example
            dec_out_pos, enc_memory_pos = inference(examples_pos, batch_size, memory_dim, seq_len, feat_dim)
            s_enc_pos = tf.slice(enc_memory_pos, [0, 0], [batch_size, split_enc])
            p_enc_pos = tf.slice(enc_memory_pos, [0, split_enc], [batch_size, memory_dim-split_enc])
            # negative example
            dec_out_neg, enc_memory_neg = inference(examples_neg, batch_size, memory_dim, seq_len, feat_dim)
            s_enc_neg = tf.slice(enc_memory_neg, [0, 0], [batch_size, split_enc])
            p_enc_neg = tf.slice(enc_memory_neg, [0, split_enc], [batch_size, memory_dim-split_enc])

            speaker_loss = tf.losses.mean_squared_error(s_enc, s_enc_neg)

        # domain-adversarial
        with tf.variable_scope('adversarial_phonetic') as scope_2:
            p_enc_adv = flip_gradient(p_enc)
            p_enc_pos_adv = flip_gradient(p_enc_pos)
            p_enc_neg_adv = flip_gradient(p_enc_neg)
            W_adv = tf.get_variable("adv_w", [2*(memory_dim-split_enc), 128])
            b_adv = tf.get_variable("adv_b", shape=[128])
            W_bin = tf.get_variable("bin_w", [128, 1])
            b_bin = tf.get_variable("bin_b", shape=[1])
            pair_pos = tf.matmul(tf.concat([p_enc_adv, p_enc_pos_adv], 1), W_adv) + b_adv
            bin_pos = tf.matmul(pair_pos, W_bin) + b_bin
            scope_2.reuse_variables()
            pair_neg = tf.matmul(tf.concat([p_enc_adv, p_enc_neg_adv], 1), W_adv) + b_adv
            bin_neg = tf.matmul(pair_neg, W_bin) + b_bin

            phonetic_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(bin_pos), logits=bin_pos) \
                          + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(bin_pos), logits=bin_neg)

        # calculate loss
        total_loss = loss(dec_out, examples, seq_len, batch_size, feat_dim) + speaker_loss + phonetic_loss
        ########
        # TODO/#
        ########

        ### learning rate decay ###
        learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("learning rate", learning_rate)

        # build a graph that grains the model with one batch of examples and
        # updates the model parameters
        train_op = train_opt(total_loss, learning_rate, 0.9)
        
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
        feed_lr = INITIAL_LEARNING_RATE*pow(LEARNING_RATE_DECAY_FACTOR,int(floor(global_step/NUM_EPOCHS_PER_DECAY)))
        ### start training ###
        for step in range(global_step, MAX_STEP):
            try:
                
                start_time = time.time()
                _, loss_value= sess.run([train_op,
                    total_loss],feed_dict={learning_rate: feed_lr})
                
                duration = time.time() - start_time
                example_per_sec = batch_size / duration
                epoch = floor(batch_size * step / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

                format_str = ('%s: epoch %d, step %d, LR %.5f, loss = %.2f ( %.1f examples/sec;'
                    ' %.3f sec/batch)')
                
                print (format_str % (datetime.now(), epoch, step, feed_lr, loss_value,
                    example_per_sec, float(duration)), end='\r')
                
                # create time line #
                #num_examples_per_step = batch_size
                #tl = timeline.Timeline(run_metadata.step_stats)
                #ctf = tl.generate_chrome_trace_format(show_memory=True)
                if step % 2000 == 0:
                    ckpt = model_file + '/model.ckpt'
                    summary_str = sess.run(summary_op,feed_dict={learning_rate:
                        feed_lr})
                    saver.save(sess, ckpt, global_step=step)
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()
                    #with open('timeline_'+str(step)+'.json','w') as f:
                    #    f.write(ctf)
                if step % NUM_EPOCHS_PER_DECAY == NUM_EPOCHS_PER_DECAY -1 :
                    feed_lr *= LEARNING_RATE_DECAY_FACTOR
            except tf.errors.OutOfRangeError:
                break
        coord.request_stop()
        coord.join(threads)
        summary_writer.flush()
    return

def test_feed(fn_list, batch_size, memory_dim, seq_len=50, feat_dim=39, split_enc=50):
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
    parser.add_argument('--batch_size',type=int, default=500,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--max_step',type=int, default=80000,
        metavar='--<max step for training>',
        help='The max step for training')
    parser.add_argument('--split_enc', type=int, default=50,
        metavar='splitting size of the encoded vector')

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
    train(fn_list, FLAG.batch_size, FLAG.hidden_dim, FLAG.split_enc)
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


