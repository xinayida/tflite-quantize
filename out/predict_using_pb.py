import tensorflow as tf
import numpy as np

def predict_using_pb(pb_path, numpy_input):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            input_tensor = sess.graph.get_tensor_by_name('input:0')
            output_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
            out = sess.run(output_tensor, feed_dict={input_tensor: numpy_input})
            return out


pb_path = 'quantize_model.pb'
# pb_path = 'converted_model.tflite'
numpy_input = np.zeros((1, 784))
#(1, 64, 48, 17)
out = predict_using_pb(pb_path, numpy_input)
print(out)
