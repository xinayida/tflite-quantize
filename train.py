
# graph = ops.Graph()
# with graph.as_default():
#   input_tensor = array_ops.zeros((1, 2, 3, 4))

# for op in (op for op in graph.get_operations()):
#     for op_input in op.inputs:
#         print(op_input)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

import os

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#隔多少步打印
display_step = 1
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
#权值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#添加伪量化结点
g = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g,
                                          quant_delay=2000000)

saver = tf.train.Saver(tf.global_variables())

#使用梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))#argmax返回一维张量中最大值所在位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

export_dir = "/Users/ww/workspace/python/deeplearning/test/quantize/model/checkpoints"
# eval_graph_file = "/Users/ww/workspace/python/deeplearning/test/quantize/model/eval_graph.pb"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        avg_cost = 0.
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            c = sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})
        # 每一轮打印损失
        # if (epoch+1) % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Itr " + str(epoch) + " accuracy: " + str(acc))
        if epoch%5 == 0:
            checkpoint_path = os.path.join(export_dir, 'fc.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)
    # tf.saved_model.simple_save(sess, export_dir, inputs={"inputs": x}, outputs={"outputs": y})
# g = tf.get_default_graph()
# with ops.Graph().as_default() as g:
#     for op in g.get_operations():
#          print(op.name,op.type) 

# tf.contrib.quantize.create_eval_graph(input_graph=g)
# acc = accuracy.eval({x:mnist.test.images,y:mnist.test.labels})
# print("acc:", acc)
# tf.saved_model.simple_save(sess, export_dir, inputs={"x": x}, outputs={"y": y})


