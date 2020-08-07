"""
基于Paddle实现skip thoughts算法
"""

import re
from collections import Counter
import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.contrib.layers import basic_gru
from conditional_gru import conditional_gru
from lineartrans import VecLinearTrans
from vector_load import vec_load
from lookup import EnVectorizer


def clean_str(string):
    """
    将文本中的特定字符串做修改和替换处理
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9:(),\.!?\'\`]", " ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class SentPre:
    """
    预处理文本，并统计文本中的句子信息，同时将词语标号，转换句子。构建bef,target,aft三个矩阵，表示相邻的三个句子，
    作为skip thoughts训练的输入。
    sent_len: 句子的长度，每个句子将被强制拉伸到这个长度，不足的将用“0”不全，多余的将被截断。
    """
    def __init__(self, sent_len):
        self.sent_len = sent_len

    def fit(self, text_or_path, encoding='utf-8', is_string=False):
        """
        读入文本，并将文本处理成句子矩阵的形式,以生成器的形式返回
        :param is_string: 输入是否为string格式，如果不是，则判断为文件地址
        :param encoding: 文件的编码形式
        :param text_or_path: 文本地址
        :return:
        """
        if not is_string:
            f = open(text_or_path, 'r', encoding=encoding)
        else:
            f = open('cache', 'w', encoding=encoding)
            f.writelines(text_or_path)
            f.close()
            f = open('cache', 'r', encoding=encoding)
        vocab_dict = {}
        full_text = ''
        for line in f.readlines():
            line = clean_str(line).lower()
            full_text += ' ' + line
            for w in line.split():
                if w in vocab_dict.keys():
                    vocab_dict[w] += 1
                else:
                    vocab_dict[w] = 1
        self.vocab_inv = [w for w, _ in Counter(vocab_dict).most_common()]
        self.vocab = {w: ind + 1 for ind, w in enumerate(self.vocab_inv)}
        f.close()
        return full_text

    def data_iters(self, full_text, for_test=False):
        """
        读入文本，并将文本处理成句子矩阵的形式,以生成器的形式返回
        :param full_text: 需要训练的字符串
        :param for_test: 是否用于测试
        :return:
        """
        sents = full_text.split('.')
        sents = [s for s in sents if len(s) > 0]

        def reader():
            for ind, line in enumerate(sents):
                if not for_test:
                    if ind > 1:
                        bef_ = [self.vocab[w] for w in sents[ind-2].split() + ['.']]
                        bef_ += [0] * self.sent_len
                        target_ = [self.vocab[w] for w in sents[ind-1].split() + ['.']]
                        target_ += [0] * self.sent_len
                        aft_ = [self.vocab[w] for w in line.split() + ['.']]
                        aft_ += [0] * self.sent_len
                        yield tuple(bef_[:self.sent_len]), tuple(target_[:self.sent_len]), tuple(aft_[:self.sent_len])
                else:
                    target_ = [self.vocab[w] for w in sents[ind-1].split() + ['.']]
                    target_ += [0] * self.sent_len
                    yield tuple(target_[:self.sent_len])
        return reader


class SkipThoughts(SentPre):
    """
    基于Paddle实现Skip Thought算法
    sent_emb_dim: 句向量的向量维度。也即encoder的隐含层尺寸。
    word_emb_dim: 词向量的向量维度。
    sent_len: 句子的长度。每个句子将被拉伸到这个长度，短的句子后边填充0，长的句子则被截断。
    lr: 训练速率
    bidirectional: 是否考虑句子的反方向，即从后向前训练，得到反向隐含层和正想隐含层拼合后作为最终的隐含层，这个参数主要控制
                   encoder层。decoder层也有这个功能，其作用未经证明。
    dropout_prob: dropout的比率
    use_gpu: 是否使用gpu
    emb_size_ratio: 不要轻易更改，为Embedding层预留足够的空间接收新单词，这个量是控制设定Embedding单词数是词汇量的多少倍
    """
    def __init__(self, sent_emb_dim,
                 word_emb_dim,
                 sent_len,
                 lr=0.001,
                 bidirectional=False,
                 dropout_prob=None,
                 num_layers=1,
                 use_gpu=True,
                 emb_size_ratio=1.5
                 ):
        super(SkipThoughts, self).__init__(sent_len=sent_len)
        self.sent_emb_dim = sent_emb_dim
        self.word_emb_dim = word_emb_dim
        self.lr = lr
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.dataloader_places = fluid.cuda_places() if use_gpu else fluid.cpu_places(8)
        self.built = False
        self.test_emb_pin = 0
        self.init = False
        self.test_fitted = False
        self.emb_size_ratio = emb_size_ratio

    def forward(self, target):
        """
        定义前向传播的逻辑
        :param target:
        :return:
        """
        if not self.built:
            self.init_emb(for_test=False)

        target_emb = self.embedding(target)

        if self.dropout_prob is not None and self.dropout_prob > 0.0:
            target_emb = fluid.layers.dropout(target_emb, dropout_prob=self.dropout_prob,
                                              dropout_implementation="upscale_in_train")

        rnn_out, last_encode_hidden = basic_gru(target_emb, None, hidden_size=self.sent_emb_dim,
                                                 num_layers=self.num_layers, bidirectional=self.bidirectional)
        #layers.assign(input=last_encode_hidden, output=self.init_encode_hidden)
        return rnn_out, last_encode_hidden

    def sent_pred(self, target, dir, encode_hidden, for_test=False):
        """
        :param target: 中心句子
        :param encode_hidden: 训练得到的编码器的隐含层
        :param for_test: 是否用于测试，这个选项决定了词嵌入层的选择
        :param dir: 预测句子所处方位，before表示预测句子前向
        :return:
        """
        if not for_test:
            target_emb = self.embedding(target)
        else:
            target_emb = self.test_embedding(target)
        if self.bidirectional:
            encode_hidden_size = self.sent_emb_dim * 2
        else:
            encode_hidden_size = self.sent_emb_dim
        if dir == 'before':
            decode_out, last_decode_hidden, all_hidden = conditional_gru(target_emb, encode_hidden, None,
                                                                         encode_hidden_size, self.word_emb_dim,
                                                                         num_layers=self.num_layers, dtype='float32',
                                                                         name='decoder_before')
            return all_hidden

        elif dir == 'after':
            decode_out, last_decode_hidden, all_hidden = conditional_gru(target_emb, encode_hidden, None,
                                                                         encode_hidden_size, self.word_emb_dim,
                                                                         num_layers=self.num_layers, dtype='float32',
                                                                         name='decoder_after')
            return all_hidden
        else:
            raise ValueError("Not support 'dir': %s." % dir)

    def init_emb(self, for_test=False):
        """
        初始化Embedding层的参数
        :param for_test: 是否用于训练
        :return:
        """
        if not for_test:
            self.embedding = fluid.Embedding(size=[int(len(self.vocab) * self.emb_size_ratio), self.word_emb_dim],
                                             padding_idx=0,
                                             param_attr=fluid.ParamAttr(name='embedding',
                                                                        initializer=fluid.initializer.UniformInitializer(
                                                                            low=-0.1, high=0.1
                                                                        ),
                                                                        learning_rate=self.lr,
                                                                        trainable=True),
                                             dtype='float32')
        else:
            self.test_emb_pin += 1
            if len(self.extra_emb) > 0:
                extra_vecs = np.array(self.extra_emb)
                extend_vecs = np.concatenate((self.emb_numpy[np.arange(len(self.vocab) + 1), :], extra_vecs), axis=0)
                extend_vecs = np.asarray(extend_vecs, dtype='float32')
            else:
                extend_vecs = self.emb_numpy[np.arange(len(self.vocab) + 1), :]
            padding_len = int(len(self.vocab) * self.emb_size_ratio) - extend_vecs.shape[0]
            paddings = np.zeros((padding_len, self.word_emb_dim), dtype='float32')
            extend_vecs = np.concatenate((extend_vecs, paddings), axis=0)
            init = fluid.ParamAttr(name='embedding_',
                                   initializer=fluid.initializer.NumpyArrayInitializer(extend_vecs),
                                   trainable=False)
            self.test_embedding = fluid.Embedding(size=[int(len(self.vocab) * self.emb_size_ratio), self.word_emb_dim],
                                                  padding_idx=0, param_attr=init, dtype='float32')

    def vec_trans(self, vec_dict_name, path, dim, epochs=5, batch_size=32, verbose_int=10, use_gpu=True):
        """
        训练一个线性转换模型，用来转换向量，以解决模型训练中出现的单词缺失（出现在测试阶段，因为训练阶段没有遇到该单词）
        :param use_gpu: 是否使用GPU训练
        :param vec_dict_name: 待查询字典名称
        :param path: 待查询字典路径
        :param dim:  待查询字典中词向量的维度
        :param epochs: 训练周期
        :param batch_size: 单次送入数据量
        :param verbose_int: 训练细节显示速度，越大显示越不频繁
        :return:
        """
        dct = vec_load(file_name=vec_dict_name, file_path=path, num=10000)
        vec_in = []
        vec_out = []
        for k in dct.keys():
            if k in self.vocab.keys():
                vec_in.append(dct[k])
                vec_out.append(self.emb_numpy[self.vocab[k]])
        self.vec_trans = VecLinearTrans(input_dim=dim, output_dim=self.word_emb_dim, use_gpu=use_gpu)
        self.vec_trans.train(vec_in, vec_out, epochs=epochs, batch_size=batch_size, verbose_int=verbose_int)
        vec_lookup = EnVectorizer()
        vec_lookup.get_path(filename=vec_dict_name, path=path, pre_path='data')
        extra_emb = vec_lookup.lookup(self.extra_vocab_inv, skip=False)
        extra_emb = self.vec_trans.trans(extra_emb)
        extra_emb = np.array(extra_emb, dtype='float32')
        self.extra_emb = np.reshape(extra_emb, (-1, self.word_emb_dim))

    def network(self, for_test=False):
        """
        定义train_model的网络结构
        :return:
        """
        if not for_test:
            before = fluid.data(name='before_train', shape=[-1, self.sent_len], dtype='int64')
            target = fluid.data(name='target_train', shape=[-1, self.sent_len], dtype='int64')
            after = fluid.data(name='after_train', shape=[-1, self.sent_len], dtype='int64')
            # 定义数据读取工具
            reader = fluid.io.DataLoader.from_generator(feed_list=[before, target, after], capacity=64, iterable=True)
            # 前向传播
            rnn_out, encode_hidden = self.forward(target)
            pred_before = self.sent_pred(target, dir='before', encode_hidden=encode_hidden, for_test=False)
            pred_after = self.sent_pred(target, dir='after', encode_hidden=encode_hidden, for_test=False)
        else:
            before = fluid.data(name='before_test', shape=[-1, self.sent_len], dtype='int64')
            target = fluid.data(name='target_test', shape=[-1, self.sent_len], dtype='int64')
            after = fluid.data(name='after_test', shape=[-1, self.sent_len], dtype='int64')
            # 定义数据读取工具
            reader = fluid.io.DataLoader.from_generator(feed_list=[before, target, after], capacity=64, iterable=True)
            # 前向传播
            rnn_out, encode_hidden = self.forward(target)
            pred_before = self.sent_pred(target, dir='before', encode_hidden=encode_hidden, for_test=True)
            pred_after = self.sent_pred(target, dir='after', encode_hidden=encode_hidden, for_test=True)

        # 将batch_size 置为1列，为什么不是0列？0列是num_layers.
        pred_before = layers.transpose(pred_before, perm=[0, 2, 1, 3])
        pred_after = layers.transpose(pred_after, perm=[0, 2, 1, 3])
        if not for_test:
            before_emb = self.embedding(before)
            after_emb = self.embedding(after)
            vocab_emb = self.embedding.parameters()[0]
        else:
            before_emb = self.test_embedding(before)
            after_emb = self.test_embedding(after)
            vocab_emb = self.test_embedding.parameters()[0]
        #loss_before = layers.cross_entropy(pred_before, before, soft_label=False)
        #loss_after = layers.cross_entropy(pred_after, after, soft_label=False)
        vocab_emb = layers.reshape(vocab_emb, shape=[1, 1, 1, vocab_emb.shape[0], vocab_emb.shape[1]])
        new_shape = pred_before.shape[:-1] + (1, ) + pred_before.shape[-1:]
        pred_before = layers.reshape(pred_before, shape=new_shape)
        pred_after = layers.reshape(pred_after, shape=new_shape)
        prob_w_before = layers.reduce_sum(layers.elementwise_mul(pred_before, vocab_emb), dim=[0, 4])
        prob_w_after = layers.reduce_sum(layers.elementwise_mul(pred_after, vocab_emb), dim=[0, 4])
        prob_w_before = layers.reduce_sum(layers.exp(prob_w_before), dim=-1)
        prob_w_after = layers.reduce_sum(layers.exp(prob_w_after), dim=-1)
        new_shape = before_emb.shape[:-1] + (1, ) + before_emb.shape[-1:]
        before_emb = layers.reshape(before_emb, shape=new_shape)
        after_emb = layers.reshape(after_emb, shape=new_shape)
        pred_before = layers.reduce_sum(layers.elementwise_mul(pred_before, before_emb), dim=[0, 3, 4])
        pred_after = layers.reduce_sum(layers.elementwise_mul(pred_after, after_emb), dim=[0, 3, 4])
        prob_before = layers.elementwise_div(layers.exp(pred_before), prob_w_before + 1e-6)
        prob_after = layers.elementwise_div(layers.exp(pred_after), prob_w_after + 1e-6)
        loss = - layers.reduce_mean((layers.log(prob_after) + layers.log(prob_before)) / 2.0)
        return loss, reader

    def train_model(self, main_prog, startup_prog):
        """
        定义训练模型，调用train_net并进行后向传播
        :param main_prog:
        :param startup_prog:
        :return:
        """
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                loss, reader = self.network(for_test=False)
                opt = fluid.optimizer.Adam(self.lr, grad_clip=fluid.clip.GradientClipByGlobalNorm(10))
                opt.minimize(loss)
        return loss, reader

    def test_model(self, main_prog, startup_prog):
        """
        定义测试模型，调用train_net但是不进行后向传播
        :param main_prog:
        :param startup_prog:
        :return:
        """
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                loss, reader = self.network(for_test=True)
        return loss, reader

    def train(self, text, epochs=5, batch_size=32, verbose_int=10):
        """
        训练
        :param text:
        :param epochs:
        :param batch_size:
        :param verbose_int:
        :return:
        """
        train_main = fluid.Program()
        train_start = fluid.Program()

        train_loss, train_reader = self.train_model(train_main, train_start)

        if not self.init:
            # 初始化
            self.exe = fluid.Executor(self.place)
            self.exe.run(train_start)
        # compile_train_main = fluid.CompiledProgram(train_main).with_data_parallel(loss_name=train_loss.name)
        train_reader.set_sample_list_generator(fluid.io.batch(self.data_iters(text, for_test=False),
                                                              batch_size=batch_size, drop_last=False),
                                               places=self.dataloader_places)
        for epoch in range(epochs):
            batch = 0
            for data in train_reader():
                loss, self.emb_numpy = self.exe.run(train_main, feed=data, fetch_list=[train_loss.name,
                                                                            train_main.all_parameters()[0]])
                batch += 1
                if batch % verbose_int == 0:
                    print('- epoch: {} -- batch: {} -- loss: {:.5f}'.format(epoch + 1, batch, loss[0]))

    def val(self, text, batch_size=32, verbose_int=10):
        if not self.test_fitted:
            raise NotImplementedError("fit_test_data should firstly be implemented.")
        test_main = fluid.Program()
        test_start = fluid.Program()

        self.init_emb(for_test=True)
        test_loss, test_reader = self.test_model(test_main, test_start)

        self.exe.run(test_start)

        test_reader.set_sample_list_generator(fluid.io.batch(self.data_iters(text, for_test=False),
                                                             batch_size=batch_size, drop_last=False),
                                              places=self.dataloader_places)
        batch = 0
        for data in test_reader():
            loss, = self.exe.run(test_main, feed=data, fetch_list=[test_loss.name])
            batch += 1
            if batch % verbose_int == 0:
                print('- batch: {} -- loss: {:.5f}.'.format(batch, loss[0]))

    def fit_test_data(self, test_data):
        """
        用来提取test_data中的所有单词，并将train_data中不存在的单词重新编号，生成一个拓展的单词词表
        :param test_data: 字符串
        :return: 返回test_data中多出单词的数量
        """
        extra_vocab_dict = {}
        max_label = np.max([i for i in self.vocab.values()])
        for line in [test_data]:
            try:
                line = clean_str(line)
                line = line.split()
            except:
                pass
            for w in line:
                if w not in self.vocab:
                    if w not in extra_vocab_dict:
                        extra_vocab_dict[w] = 1
                    else:
                        extra_vocab_dict[w] += 1
        if len(extra_vocab_dict) == 0:
            self.extra_vocab_inv = ['<nomore>']
            self.extra_vocab = {w: max_label + ind + 1 for ind, w in enumerate(self.extra_vocab_inv)}
            self.test_fitted = True
            for w, ind in self.extra_vocab.items():
                self.vocab[w] = ind
            return 0
        self.extra_vocab_inv = [w for w, _ in Counter(extra_vocab_dict).most_common()]
        self.extra_vocab = {w: max_label + ind + 1 for ind, w in enumerate(self.extra_vocab_inv)}
        for w, ind in self.extra_vocab.items():
            self.vocab[w] = ind
        self.test_fitted = True
        return len(self.extra_vocab_inv)


if __name__ == '__main__':
    '''from paddle import batch
    import time
    sentpre = SentPre(5)
    reader = sentpre.fit_data_iters('Twilight01wilight.txt')

    pin = 0
    start = time.time()
    for i in batch(reader, 2)():
        print(np.array(i))
        #a, b, c = i
        pin += 1
        if pin >= 1:
            break
    print(time.time() - start)'''

    st = SkipThoughts(sent_emb_dim=1200, word_emb_dim=100, sent_len=20, num_layers=2, use_gpu=True)
    full_text = st.fit('Twilight01wilight.txt', encoding='utf-8', is_string=False)
    st.train(full_text[:1001], epochs=1, batch_size=2, verbose_int=1)
    st.fit_test_data(full_text[1001:1500])
    st.vec_trans('vectors.txt', path='D:/OneDrive/WORK/Python projects/Reproduce Papers/Skip-Thoughts', dim=300)
    st.val(full_text[1001:1500], batch_size=2, verbose_int=1)
