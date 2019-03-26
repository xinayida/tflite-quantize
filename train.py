
import tensorflow as tf
import models
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

import os

root = os.getcwd()

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#隔多少步打印
display_step = 1
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# #定义placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

logits = models.create_mode(x)


#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))

#添加伪量化结点
g = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g,
                                          quant_delay=2000000)
#存储模型工具
saver = tf.train.Saver(tf.global_variables())

#使用梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(logits,1))#argmax返回一维张量中最大值所在位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

export_dir = os.path.join(root, "model/checkpoints")

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(7):
        avg_cost = 0.
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Itr " + str(epoch) + " accuracy: " + str(acc))
        if epoch%2 == 0:#记录训练中模型参数
            checkpoint_path = os.path.join(export_dir, 'fc.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)
# tf.saved_model.simple_save(sess, export_dir, inputs={"inputs": x}, outputs={"outputs": y})
# g = tf.get_default_graph()
# with ops.Graph().as_default() as g:
#     for op in g.get_operations():
#          print(op.name,op.type) 


