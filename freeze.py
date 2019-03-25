
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import models
import argparse
import os
import sys

from tensorflow.python.framework import graph_util

root = os.getcwd()

output_file = os.path.join(root,"out/quantize_model.pb")
checkpoints = os.path.join(root,"model/checkpoints")
out_dir = os.path.dirname(output_file)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def main():
    sess = tf.InteractiveSession()
    input =  tf.placeholder(tf.float32, [1, 784], name='input')
    logits = models.create_mode(input)
    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')
    tf.contrib.quantize.create_eval_graph()
    models.load_variables_from_checkpoint(sess,checkpoints)
    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax'])
    tf.train.write_graph(
        frozen_graph_def,
        out_dir,
        os.path.basename(output_file),
        as_text=False)
    tf.logging.info('Saved frozen graph to %s', output_file)

if __name__ == "__main__":
    main()