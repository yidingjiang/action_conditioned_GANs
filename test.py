import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
import random

from util import *
from ops import *
from models import *
from train import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('input_frame_path', type=str)
    parser.add_argument('input_action_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--dna', action='store_true')

    sess = tf.InteractiveSession()
    queue_runners = tf.train.start_queue_runners(sess)
    trainer = Trainer(sess, True, 'bce', 'adam', parser.dna)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(parser.model_path))
    
    seq = np.load(parser.input_frame_path)
    act = np.load(parser.input_action_path)

    seq_batch = seq[32:96]
    act_batch = act[32:96]

    test_g_out, gt = trainer.test_sequence(seq_batch[:,6:], 
                                           seq_batch[:,6:], 
                                           act_batch[:,6:])
    for i in range(test_g_out.shape[1]):
        save_samples(parser.output_path, 
                     seq_batch[:,[6 + (i+1)*2 for i in range(6)]], 
                     test_g_out, 
                     np.array([0]), 
                     i, 
                     gif=True)
