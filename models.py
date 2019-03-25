from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

def create_mode(input, is_training):
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(input, W) + b
    return logits

def load_variables_from_checkpoint(sess, path):
    saver = tf.train.Saver(tf.global_variables())
    latest_checkpoint = tf.train.latest_checkpoint(path)
    saver.restore(sess, latest_checkpoint)