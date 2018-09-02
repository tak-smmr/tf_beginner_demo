# import
# -*- coding:utf-8 -*-
import tensorflow as tf
# 計算グラフ
max_time = 50
input_size = 10

# 入力データ
x = tf.placeholder(tf.float32, [None, max_time, input_size])

### 普通のRNNのcellを定義
#cell = tf.nn.rnn_cell.BasicRNNCell(num_units=100, activation=tf.nn.relu)

### LSTMのcellを定義
cell = tf.nn.rnn_cell.LSTMCell(num_units=100, use_peepholes=True)

### MultiRNN
#cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=100, activation=tf.nn.relu)
# LSTMのcellを定義
#cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=100, use_peepholes=True)
# ドロップアウトをcell_2に負荷
#cell2 = tf.nn.rnn_cell.DropoutWrapper(cell_2, output_keep_prob=0.6)
# cell_1, cell_2の順番になる多層のrnn_cell
#cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_1, cell2])

# 時間展開して出力を取得
outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)

# 3階テンソルを２階テンソルのリストに変換
outputs_list = tf.unstack(outputs, axis=1)

# 最終時系列情報を取得
last_output = outputs_list[-1]

w = tf.Variable(tf.truncated_normal([100,10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
out = tf.nn.softmax(tf.matmul(last_output, w ) + b)
