import numpy as np
import time
import os
import argparse
from solver import Solver

FLAG = None

def addParser():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Testing Script')
    parser.add_argument('--init_lr',  type=float, default=0.1,
        metavar='<--initial learning rate>')
    parser.add_argument('--p_hidden_dim',type=int, default=128,
        metavar='<--phonetic hidden dimension>',
        help='The hidden dimension of a neuron')
    parser.add_argument('--s_hidden_dim',type=int, default=128,
        metavar='<--speaker hidden dimension>',
        help='The hidden dimension of a neuron')
    parser.add_argument('--batch_size',type=int, default=32,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--seq_len',type=int, default=50,
        metavar='--<seq len>',
        help='length of a sequence')
    parser.add_argument('--feat_dim',type=int, default=39,
        metavar='--<feat dim>',
        help='feature dimension')
    parser.add_argument('--stack_num',type=int, default=3,
        metavar='--<number of rnn stacks>',
        help='number of rnn stacks')

    parser.add_argument('log_dir', 
        metavar='<log directory>')
    parser.add_argument('model_dir', 
        metavar='<model directory>')
    parser.add_argument('test_feat_scp', 
        metavar='<testing feature scp file>')    
    parser.add_argument('feat_dir', 
        metavar='<feature directory>')    
    parser.add_argument('model_type', 
        metavar='<model type>')    
    parser.add_argument('word_word_dir', 
        metavar='<word word bottleneck directory>')    
    parser.add_argument('word_spk_dir', 
        metavar='<word spk bottleneck directory>')    
    parser.add_argument('spk_word_dir', 
        metavar='<spk word bottleneck directory>')    
    parser.add_argument('spk_spk_dir', 
        metavar='<spk spk bottleneck directory>')    
    parser.add_argument('phonetic_file', 
        metavar='<phonetic features>')    
    return parser

def main():
    solver = Solver(FLAG.model_type, FLAG.stack_num, FLAG.feat_dir, None, FLAG.test_feat_scp, FLAG.batch_size,
                    FLAG.seq_len, FLAG.feat_dim, FLAG.p_hidden_dim, FLAG.s_hidden_dim, FLAG.init_lr,
                    FLAG.log_dir, FLAG.model_dir, None)
    print ("Solver constructed!")
    solver.test(FLAG.word_word_dir, FLAG.word_spk_dir, FLAG.spk_word_dir, FLAG.spk_spk_dir, FLAG.phonetic_file)

if __name__ == '__main__':
    parser = addParser()
    FLAG = parser.parse_args()
    main()


