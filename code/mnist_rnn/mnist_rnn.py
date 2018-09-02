# import
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# データ読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 計算グラフ
#入力データ整形
num_seq = 28
num_input = 28
x = tf.placeholder(tf.float32, [None, 784])
input = tf.reshape(x, [-1, num_seq, num_input])

y = tf.placeholder(tf.float32, [None, 10])
#ユニット数128個のLSTMセル
#三段に積む
stacked_cells = []
for i in range(3):
    stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)
#dynamic_rnn構築
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32, time_major=False)

# ミニマムはこれで良い
#cell = tf.nn.rnn_cell.LSTMCell(num_units=128, use_peepholes=True)
#outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32, time_major=False)

# 3階テンソルを2階テンソルのリストに変換
outputs_list = tf.unstack(outputs, axis=1)
# 最終時系列情報を取得
last_output = outputs_list[-1]

#出力層
w = tf.Variable(tf.truncated_normal([128,10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax(tf.matmul(last_output, w ) + b)

# 損失関数
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

# 訓練方法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 評価/検証用
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 訓練
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

    #10回に1回精度を検証
    step = i+1
    if step % 10 == 0:
        acc_val = sess.run(accuracy ,feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
        print('Step %d: accuracy = %.2f' % (step, acc_val))

# テストデータで評価
print("正解率 : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
