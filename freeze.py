
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import models
import argparse
import os.path
import sys

from tensorflow.python.framework import graph_util

FLAGS = {'output_file': '/Users/ww/workspace/python/deeplearning/test/quantize/out/quantize_model.pb','start_checkpoint': '/Users/ww/workspace/python/deeplearning/test/quantize/variables'}

def main(_):
    sess = tf.InteractiveSession()
    input =  tf.placeholder(tf.float32, [1, 784], name='input')
    logits = models.create_mode(input, False)
    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')
    tf.contrib.quantize.create_eval_graph()
    check_point = "/Users/ww/workspace/python/deeplearning/test/quantize/model"
    models.load_variables_from_checkpoint(sess,check_point) #(FLAGS.start_checkpoint)
    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax'])
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(FLAGS['output_file']),
        os.path.basename(FLAGS['output_file']),
        as_text=False)
    tf.logging.info('Saved frozen graph to %s', FLAGS['output_file'])

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--start_checkpoint',
    #     type=str,
    #     default='',
    #     help='If specified, restore this pretrained model before any training.')
    # parser.add_argument(
    #     '--output_file', type=str, help='Where to save the frozen graph.')
    # FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main)