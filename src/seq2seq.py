from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip 

import copy
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import dtypes 
from tensorflow.python.ops import variable_scope

def basic_rnn_seq2seq_with_bottle_memory(encoder_inputs,
                                         decoder_inputs,
                                         cell,
                                         dtype=dtypes.float32,
                                         scope=None):
    """Basic RNN sequence-to-sequence model. 

    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size]
      decoder_inputs: A list of 2D Tensors [batch_size x input_size]
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype of the initial state of the RNN cell (default:
        tf.float32).
      scope: VariableScope for the created subgraph; default: "rnn_seq2seq_BN"

    Returns:
      
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.

      enc_state: The state of each encoder cell in the final time-step.
          This is a 2D Tensor of shape [batch_size x cell.state_size]

      dec_state: The state of each decoder cell in the final time-step.
          This is a 2D Tensor of shape [batch_size x cell.state_size]
    """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
        _, enc_state = core_rnn.static_rnn(cell, encoder_inputs, dtype=dtype)
        outputs, dec_state = seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)

        return outputs, enc_state, dec_state


def stack_rnn_seq2seq_with_bottle_memory(encoder_inputs,
                                         decoder_inputs,
                                         cell,
                                         stack_num=3,
                                         dtype=dtypes.float32,
                                         scope=None):
    """Stacking RNN seq2seq model with bottleneck.
    
    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size] 
      decoder_inputs: A list of 2D Tensors [batch_size x input_size]
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      stack_num: the number to stack in seq2seq model 
      dtype: The dtype of the initial state of the RNN cell (default:
        tf.float32)
      
    Returns:
      outputs: A list of the same length as decoer_inputs of 2D Tensors with 
        shape [batch_size x output_size] containing the generated outputs.
      enc_state: The state of each encoder cell in the final time_step.
        This is a 2D Tensor of shape [batch_size x cell.state_size]
      dec_state: The state of each decoder cell in the final time-step.
        This is a 2D Tensor of shape [batch_size x cell.state_size]
    """
    with variable_scope.variable_scope(scopre or "stack_rnn_enc_1"):
        enc_cell = copy.copy(cell)
        enc_output, enc_state = core_rnn.static_rnn(enc_cell, encoder_inputs,
            dtype=dtype)
    for i in range(stack_num):
        with variable_scope.variable_scope(scope or "stack_rnn_encoder_"+str(i)):
            enc_cell = copy.copy(cell)
            enc_output, enc_state = core_rnn.static_rnn(enc_cell,enc_output,dtype=dtype)

    with variable_scope.variable_scope(scope or "stack_rnn_dec_0"):
        dec_cell = copy.copy(cell)
        dec_output, dec_state = seq2seq.rnn_decoder(decoder_inputs,
            enc_state,dec_cell)
    for i in range(stack_num):
        with variable_scope.variable_scope(scope or
            "stack_rnn_decoder_"+str(i)):
            dec_cell = copy.copy(cell)
            dec_output, dec_state = core_rnn.static_rnn(dec_cell, dec_output,
                dtype=dtype)

    return dec_output, enc_state, dec_state



