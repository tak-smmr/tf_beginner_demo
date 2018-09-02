# import
# -*- coding:utf-8 -*-

import tensorflow as tf
import math
import argparse

# 引き数処理
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data_w2v/', help='Data set directory.')
parser.add_argument('--log_dir', type=str, default='logs_w2v/', help='Log directory.')
parser.add_argument('--max_vocab', type=int, default=2000, help='Max Vocablary size.')
parser.add_argument('--skip_window', type=int, default=2, help='How many words to consider left and right.')
parser.add_argument('--num_skips', type=int, default=4, help='How many times to reuse an input to generate a label.')
parser.add_argument('--embedding_size', type=int, default=64, help="Dimension of the embedding vector.")
parser.add_argument('--num_sumpled', type=int, default=64, help="Number of negative examples to sample." )
parser.add_argument('--num_step', type=int, default=10000, help="Train step." )
parser.add_argument('--batch_size', type=int, default=64, help="Batch size." )
parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate." )
parser.add_argument('--create_tsv', type=bool, default=True, help="Create words.tsv or not." )

FLAGS, unparsed = parser.parse_known_args()

# DataSetクラス
import glob
import re
import collections
import random

import numpy as np
from janome.tokenizer import Tokenizer

class DataSet(object):

    def __init__(self, data_dir, max_vocab):

        #全データセットのファイルパスを取得
        file_pathes = []
        for file_path in glob.glob(data_dir+'*'):
            file_pathes.append(file_path)

        #ファイルを読み込み
        row_documents = [self._read_docment(file_path) for file_path in file_pathes]
        #必要な部分だけ抽出
        documents = [self._preprocessing(document) for document in row_documents]
        #形態素解析
        splited_documents = [self._morphological(document) for document in documents]

        words = []
        for word_list in splited_documents:
            words.extend(word_list)

        #データセット作成
        self.id_sequence, self.word_frequency, self.w_to_id, self.id_to_w = self._build_data_sets(words, max_vocab)
        print('Most common words (+UNK)', self.word_frequency[:5])
        print('Sample data.')
        print(self.id_sequence[:10])
        print([self.id_to_w[i] for i in self.id_sequence[:10]])
        self.data_index = 0

    #ファイルの読み込み
    def _read_docment(self, file_path):
        with open(file_path, 'r', encoding='sjis') as f:
            return f.read()

    #ヘッダなどの不要データを前処理。必要な部分だけを返す。
    def _preprocessing(self, document):

        lines = document.splitlines()
        processed_line = []

        horizontal_count = 0

        for line in lines:

            #ヘッダーは読み飛ばす
            if horizontal_count < 2:
                if line.startswith('-------'):
                    horizontal_count += 1
                continue
            #フッターに入る行になったらそれ以降は無視
            if line.startswith('底本：'):
                break

            line = re.sub(r'《.*》', '', line) #ルビを除去
            line = re.sub(r'［.*］', '', line) #脚注を除去
            line =re.sub(r'[!-~]', '', line) #半角記号を除去
            line =re.sub(r'[︰-＠]', '', line) #全角記号を除去
            line = re.sub('｜', '', line) # 脚注の始まりを除去

            processed_line.append(line)

        return ''.join(processed_line)

    #形態素解析
    def _morphological(self, document):

        word_list = []
        t = Tokenizer()
        for token in t.tokenize(document):
            #名詞（一般）動詞（自立）、形容詞（自立）以外は除外
            if token.part_of_speech.startswith('名詞,一般') and token.base_form != '':
                word_list.append(token.base_form)
            if token.part_of_speech.startswith('動詞,自立') and token.base_form != '':
                word_list.append(token.base_form)
            if token.part_of_speech.startswith('形容詞,自立') and token.base_form != '':
                word_list.append(token.base_form)
        return word_list

    #辞書作成
    def _build_data_sets(self, words, max_vocab):

        #単語出現回数を解析。出現数が少ないたんをUnknown wordとしてひとくくりに扱う
        word_frequency = [['UNW', -1]]
        word_frequency.extend(collections.Counter(words).most_common(max_vocab - 1))
        #単語=>IDの辞書
        w_to_id = dict()
        for word, _ in word_frequency:
            w_to_id[word] = len(w_to_id)
        #形態素解析した文章を単語IDの並びに変換
        id_sequence = list()
        unw_count = 0
        for word in words:
            #UNK処理
            if word in w_to_id:
                index = w_to_id[word]
            else:
                index = 0
                unw_count += 1
            id_sequence.append(index)
        word_frequency[0][1] = unw_count
        #単語ID=>単語の辞書
        id_to_w = dict(zip(w_to_id.values(), w_to_id.keys()))
        return id_sequence, word_frequency, w_to_id, id_to_w

    # num_skip:１つの入力をどれだけ再利用するか
    # skip_window: 左右何語までを正解対象にするか
    def create_next_batch(self, batch_size, num_skips, skip_window):

        assert batch_size % num_skips == 0
        #一つの入力の再利用回数が対象範囲全件を超えてはならない
        assert num_skips <= 2 * skip_window
        inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        #データセットが1週しそうならindexを最初にもどす
        if self.data_index + span > len(self.id_sequence):
            self.data_index = 0
        #初期のqueueを構築(window内の単語をすべて格納)
        buffer.extend(self.id_sequence[self.data_index:self.data_index+span])
        self.data_index += span

        for i in range(batch_size // num_skips):
            #中心は先に正解データから除外
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                #すでに選ばれている物以外から正解データのインデックスを取得
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                #次回以降targetにならないように
                targets_to_avoid.append(target)
                #入力値になるのはbufferの中心
                inputs[i * num_skips + j] = buffer[skip_window]
                #ランダムに指定した周辺単語が正解データに
                labels[i * num_skips + j, 0] = buffer[target]

            #次に入れる単語がデータセットにない場合はbufferには最初の値を入力
            if self.data_index == len(self.id_sequence):
                buffer = self.id_sequence[:span]
                self.data_index = span
            else:
                #bufferに次の単語を追加してindexを1進める
                buffer.append(self.id_sequence[self.data_index])
                self.data_index += 1
        #最後の方のデータが使われないことを避けるために少しだけindexを元に戻す
        self.data_index = (self.data_index + len(self.id_sequence) - span) % len(self.id_sequence)

        return inputs, labels


# データセットオブジェクトを作成
data = DataSet(FLAGS.data_dir, FLAGS.max_vocab)

# Embeddings用に使うラベルをtsv形式で保存
if FLAGS.create_tsv:
    sorted_dict = sorted(data.w_to_id.items(), key=lambda x: x[1])
    words = [ "{word}\n".format(word=x[0]) for x in sorted_dict]
    tf.gfile.MakeDirs(FLAGS.log_dir)
    with open(FLAGS.log_dir+"words.tsv", 'w', encoding="utf-8") as f:
        f.writelines(words)
    print("Embeddings metadata was saved to "+FLAGS.log_dir+"/words.tsv")
batch_size = FLAGS.batch_size
embedding_size = FLAGS.embedding_size
vocab_size = len(data.w_to_id)

# placeholderの定義
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# 中間層
embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embedding, train_inputs)

# 出力層
nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev =1.0 / math.sqrt(embedding_size)))
nce_biases  = tf.Variable(tf.zeros([vocab_size]))

# 損失関数
nce_loss = tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, FLAGS.num_sumpled, vocab_size)
loss = tf.reduce_mean(nce_loss)

# 訓練方法
global_step = tf.train.get_or_create_global_step()
train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

# モデル保存用
saver = tf.train.Saver(max_to_keep=3)

# 訓練
sess = tf.Session()

# 初期化 or モデル読み込み
ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_dir)
if ckpt_state:
    last_model = ckpt_state.model_checkpoint_path
    saver.restore(sess,last_model)
    print("model was loaded:", last_model)
else:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("initialized.")

last_step = sess.run(global_step)
average_loss = 0
for i in range(FLAGS.num_step):

    step = last_step + i + 1
    batch_inputs, batch_labels = data.create_next_batch(batch_size, FLAGS.num_skips, FLAGS.skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 100 == 0:
        average_loss /= 100
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0
        saver.save(sess, FLAGS.log_dir+'my_model.ckpt', step)
