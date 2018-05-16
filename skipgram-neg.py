
from __future__ import print_function
import collections
import math
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import range
from sklearn.manifold import TSNE
import jieba
import os
import re

import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
def _readfile(path):
    '''
        文件读写函数
        :param path: 文件路径
        :return:
        '''
    with open(path, "r", encoding='gbk') as fp:
        content = fp.read()
    return content

stopwords_path = 'stopwordsRemind.txt'
stopwords = _readfile(stopwords_path).splitlines()

class MySentences(object):
    def __init__(self, dirname, stopwords):
        self.dirname = dirname
        self.stopwords = stopwords

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(os.path.join(self.dirname, fname), encoding='gbk', errors='ignore') as f:
                line = f.read()
                if len(line) < 2:
                    continue
                if re.search('[0-9]{12,14}', line) and len(line) <= 14:
                    continue
                # yield ' '.join(jieba.cut(line)).split()
                content = ' '.join([word for word in jieba.cut(line) if word not in self.stopwords])
                if re.search('[\u4e00-\u9fa5]', content):
                    # print(content)
                    yield ' '.join([word for word in jieba.cut(line) if word not in self.stopwords]).split()



def read_data(files,stopwords):
    words = []
    for fname in files:
        with open(fname, encoding='gbk', errors='ignore') as f:
            line = f.read()
            if len(line) < 2:
                continue
            if re.search('[0-9]{12,14}', line) and len(line) <= 14:
                continue
            # yield ' '.join(jieba.cut(line)).split()
            content = ' '.join([word for word in jieba.cut(line) if word not in stopwords])
            if re.search('[\u4e00-\u9fa5]', content):
                print(content)
                words.extend(' '.join([word for word in jieba.cut(line) if word not in stopwords]).split())
    return words
    # with open(path,encoding='gbk',errors='ignore') as f:
    #     line = f.read()
    #     line = line.replace('\n','').replace('\r','').strip().replace('\\s+','').replace(' ','')
    #     line = f.read()
    #     if len(line) < 2:
    #         continue
    #     if re.search('[0-9]{12,14}', line) and len(line) <= 14:
    #         continue
    #     # yield ' '.join(jieba.cut(line)).split()
    #     content = ' '.join([word for word in jieba.cut(line) if word not in self.stopwords])
    #     if re.search('[\u4e00-\u9fa5]', content):
    #
    #     print(content)
    #     words = list(jieba.cut(content))
    # return words

vocabulary_size = 50000
def build_dataset(words):
    '''
    :param words:
    :return: data(词的索引)，count （word，count）dictionary{word，index}但是含有 unk特殊字符表示不出现在字典中的值。
                                                   reverse_dictionary(index, word)  索引——词
    '''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

filepath = 'D:\9900\查件'

def read_files(filepath):
    filelist = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            filelist.append(os.path.join(root, file))
    return filelist

filelist = read_files(filepath)
words = read_data(filelist,stopwords)
# for file in filelist:
#     words.extend(read_data(file))
# print(words)
data, count, dictionary, reverse_dictionary = build_dataset(words)

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch_size = 64
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))


    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))


    optimizer = tf.train.AdagradOptimizer(0.0001).minimize(loss)


norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1,keep_dims=True))

normalized_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


num_steps = 50001
#
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    #训练过程代码
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l


        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    #返回 embeddings层的结果
    final_embeddings = normalized_embeddings.eval()


num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  plt.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)




if __name__ == '__main__':
    # words = read_data('1.txt')
    data, count, dictionary, reverse_dictionary = build_dataset(words)