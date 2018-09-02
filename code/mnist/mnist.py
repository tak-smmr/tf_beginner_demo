
# coding: utf-8

# import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# データ読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 計算グラフ
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

out = tf.nn.softmax(tf.matmul(x, W) + b)

# 損失関数
loss = tf.reduce_mean(tf.square(y - out))

# 訓練方法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 評価/検証用
correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 訓練
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    step = i+1        
    #10回に1回精度を検証
    if step % 10 == 0:
        acc_val = sess.run(accuracy ,feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
        print('Step %d: accuracy = %.2f' % (step, acc_val))         

# テストデータで評価
print("正解率 : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
