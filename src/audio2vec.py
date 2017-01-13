#!/usr/bin/env python3
import tensorflow as tf
import tempfile
import numpy as np

from tensorflow.python.ops import seq2seq, rnn_cell


### building sequence to sequence graph ###
def graph_build(batch_size, memory_dim, seq_length=200,feat_dim=39):
    
    # encoder input tensor: enc(seq_length, batch_size, feat_dim) #
    enc_inp = [tf.placeholder(tf.float32, shape=(batch_size,feat_dim), name="inp%i" %t) for t in range(seq_length)]
    # decoder output tensor: labels(seq_length, batch_size, feat_dim) #
    labels = [tf.placeholder(tf.float32, shape=(batch_size,feat_dim), name="labels%i" %t) for
        t in range(seq_length)]
    # weight tensor #
    weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels ]
    
    # Decoder input: prepend some "GO" token and drop the final
    # token of the encoder input 

    dec_inp = ([tf.zeros_like(enc_inp[0], dtype=tf.float32, name="GO")] + enc_inp[:-1])

    # Initial memory value for recurrence #
    prev_men = tf.zeros((batch_size, memory_dim))

    ### these two calss defines main cell in seq2seq and seq2seq model ###
    cell = rnn_cell.GRUCell(memory_dim)
    
    dec_outputs, dec_memory = seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, cell)
    ######################################################################

    ### build loss for the seq2seq loss ###
    ### variable: summary_op            ###
    ### project the decoder outputs through a matrix transform ###

    ### dec_outputs: [Time, batch_size, memory] ###
    ### label: [Time, batch_size, feat_dim] ###
    dec_reshape = tf.transpose(tf.reshape(dec_outputs, (seq_length*batch_size, \
        memory_dim)))
    
    W_p = tf.get_variable("output_proj_w", [feat_dim, memory_dim] )
    b_p = tf.get_variable("output_proj_b", shape=(feat_dim),   \
        initializer=tf.constant_initializer(0.0))
    b_p = [ b_p for i in range(seq_length*batch_size) ]
    b_p = tf.transpose(b_p)
    dec_proj_outputs =  tf.matmul(W_p, dec_reshape) + b_p
    ### dec_proj_output: projection from [Time, batch_size, memory_dim] to 
    ###                                  [feat_dim, Time* batch_size]
    ### labels_trans: reshape labels into [feat_dim, Time* batch_size] ### 
    labels_trans = tf.transpose(tf.reshape(labels, shape=(seq_length*batch_size, feat_dim)))
    labels_trans = tf.reshape(labels_trans, shape=[-1])
    dec_proj_outputs = tf.reshape(dec_proj_outputs, shape=[-1])
    # loss = tf.nn.l2_loss(tf.sub(dec_proj_outputs, labels_trans), name="l2loss")
    tmp_loss = tf.sub(dec_proj_outputs, labels_trans)
    tmp_loss = tf.multiply(tmp_loss, tmp_loss)
    loss = tf.divide(tmp_loss, 2)
    loss = tf.reduce_sum(loss)
    tf.scalar_summary("loss", loss)

    summary_op = tf.merge_all_summaries()
    
    return summary_op, loss, dec_outputs, dec_memory, enc_inp, labels

def train_opt(loss, learning_rate, momentum):
    ### Optimizer building              ###
    ### variable: train_op              ###
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def loggin(graph):
    logdir = tempfile.mkdtemp()
    print (logdir)
    summary_writer = tf.train.SummaryWriter(logdir, graph)
    return summary_writer

def batch_training(batch_size, feat_dim, seq_length, enc_inp, labels, train_op,
    loss,  summary_op,sess):
    ### example training ###
    X = [[ np.random.choice(2, size=(feat_dim,), replace=True)
         for _ in range(batch_size)] for j in range(seq_length)]
    Y = X[:]
    # print (np.shape(X), np.shape(Y))
    feed_dict = { enc_inp[t]: X[t] for t in range(seq_length) }
    feed_dict.update({labels[t] : Y[t] for t in range(seq_length) })

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

def dataQ_feeding(filename_queue, feat_dim, seq_length):
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
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    ### read the csv file into features ###
    record_defaults = [[1.] for i in range(feat_dim*seq_length)]
    seq = tf.decode_csv(value, record_defaults=record_defaults)
    result.mfcc = tf.reshape(seq, shape=(feat_dim , seq_length))
    
    return result
    #

def main():
    tf.reset_default_graph()
    sess = tf.Session()
    summary_op, loss, dec_outputs, dec_memory, enc_inp, labels = graph_build(10,150)
    train_op = train_opt(loss, 0.00000001, 0.9 )
    summary_writer = loggin(sess.graph)
    sess.run(tf.initialize_all_variables())
    for t in range(500):
        loss_t, summary = batch_training(10, 39, 200, enc_inp, labels,\
            train_op, loss, summary_op, sess ) 
        print (loss_t)
        summary_writer.add_summary(summary,t)
    summary_writer.flush()
if __name__ == '__main__':
    main()
