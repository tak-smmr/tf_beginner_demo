# import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# データ読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 計算グラフ
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

### 入力層から中間層
w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1))
b_1 = tf.Variable(tf.zeros([64]))
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

### 中間層から出力層
w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))
b_2 = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

# 損失関数
loss = tf.reduce_mean(tf.square(y - out))

# 訓練方法
global_step = tf.train.get_or_create_global_step()
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

# 評価/検証用
correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Saverの定義
saver = tf.train.Saver()

# 訓練
sess = tf.Session()

ckpt_state = tf.train.get_checkpoint_state('models/')
if ckpt_state:
    #読み込み処理
    last_model = ckpt_state.model_checkpoint_path
    saver.restore(sess, last_model)
else:
    #初期化
    sess.run(tf.global_variables_initializer())
last_step = sess.run(global_step)

for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

    step = last_step + i + 1
    #10回に1回精度を検証
    if step % 10 == 0:
        acc_val = sess.run(accuracy ,feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
        print('Step %d: accuracy = %.2f' % (step, acc_val))

    #100回に1回データを保存
    if step % 100 == 0:
        saver.save(sess, 'models/my_model', global_step = step)

# テストデータで評価
print ("正解率 : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
