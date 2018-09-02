# import

# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# データ読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 計算グラフ
x = tf.placeholder(tf.float32, shape=(None, 784))
img = tf.reshape(x,[-1,28,28,1])

### 畳み込み層1
# 使わない場合
w_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv = tf.Variable(tf.zeros(shape=[32]))
conv = tf.nn.conv2d(img, w_conv, strides=[1, 1, 1, 1], padding= "SAME")
conv1 = tf.nn.relu(conv + b_conv)

# 使った場合
#conv1 = tf.layers.conv2d(
#      inputs=img, # 入力するテンソル
#      filters=32, # 畳み込み後のチャンネル数
#      strides=(1, 1), # ストライド [縦方向,横方向] 
#      kernel_size=[5, 5], # フィルタのサイズ [高さ,幅] 
#      padding="same", # パディング
#      activation=tf.nn.relu # 活性化関数Relu 
#)

#プーリング層1
# 使わない場合
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 使った場合
#pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, padding="SAME")

### 畳み込み層2"
# 畳み込み層２ プーリング層2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    strides=(1, 1),
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 全結合層
#畳み込まれているものをフラットな形に変換
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# 使った場合
#dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
# 使わない場合
w_hidden = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_hidden = tf.Variable(tf.zeros(shape=[1024]))
dense = tf.nn.relu(tf.matmul(pool2_flat, w_hidden) + b_hidden)

#出力層
out = tf.layers.dense(
  inputs=dense,
  units=10,
  activation=tf.nn.softmax)

#正解データの型を定義
y = tf.placeholder(tf.float32, [None, 10])

#損失関数
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

#訓練
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#評価
correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 訓練
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 学習
for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(50)
    sess.run( train_step, feed_dict={x: train_images, y: train_labels})

    step = i+1        
    if step % 10 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        print('Step %d: accuracy = %.2f\tloss = %.2f' % (step, acc_val, loss_val))


# テストデータで評価
print("正解率 : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
