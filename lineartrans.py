"""
基于Paddle，训练一个线性转换的参数W，完成两个向量之间的线性转换Vx = WVy
"""
from paddle import fluid
import time
import numpy as np


class VecLinearTrans:
    """
    input_dim: 输入向量的维度
    output_dim: 输出向量的维度
    use_gpu: 是否使用gpu进行训练
    """
    def __init__(self, input_dim, output_dim, use_gpu=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = fluid.layers.create_parameter([input_dim, output_dim], dtype='float32', name='W')
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.deployed = False

    def forward(self, vec_in):
        out = fluid.layers.matmul(vec_in, self.W)
        return out

    def get_batch(self, x, y=None, batch_size=32):
        len_x = len(x)
        batch_num = int(np.ceil(len_x / batch_size))
        for batch_id in range(batch_num):
            batch_x = x[batch_id*batch_size:(batch_id+1)*batch_size]
            if y is not None:
                batch_y = y[batch_id*batch_size:(batch_id+1)*batch_size]
                yield batch_x, batch_y, batch_id+1, batch_num
            else:
                yield batch_x, batch_id+1, batch_num

    def train(self, vec_in, vec_out, epochs=5, batch_size=32, verbose_int=100):
        if not self.deployed:
            vec_1 = fluid.data(name='vec_in', shape=(-1, self.input_dim), dtype='float32')
            vec_2 = fluid.data(name='vec_out', shape=(-1, self.output_dim), dtype='float32')

            self.main_pg = fluid.default_main_program()
            start_pg = fluid.default_startup_program()

            self.pred = self.forward(vec_1)
            self.loss = fluid.layers.reduce_sum(fluid.layers.elementwise_sub(self.pred, vec_2) ** 2)

            self.test_pg = self.main_pg.clone(for_test=True)

            opt = fluid.optimizer.Adam()
            opt.minimize(self.loss)

            self.exe = fluid.Executor(self.place)
            self.exe.run(start_pg)

            self.deployed = True
        start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            len_batch = 0
            ave_loss = 0.0
            for vec_in_, vec_out_, batch_id, batch_num in self.get_batch(vec_in, vec_out, batch_size=batch_size):
                len_ = len(vec_in_)
                vec_in_ = np.asarray(vec_in_, dtype='float32')
                vec_out_ = np.asarray(vec_out_, dtype='float32')
                feed_list = {'vec_in': vec_in_, 'vec_out': vec_out_}
                loss, = self.exe.run(self.main_pg, feed=feed_list, fetch_list=[self.loss])
                ave_loss = (loss[0] * len_ + ave_loss * len_batch) / (len_batch + len_)
                len_batch += len_
                if batch_id % verbose_int == 0:
                    print("{}/{} epochs - {}/{} batches - ETA {:.0f}s - loss: {:.4f} ...".format(
                        str(epoch + 1).rjust(len(str(epochs))),
                        epochs, str(batch_id).rjust(len(str(batch_num))), batch_num,
                        (time.time() - epoch_start) / batch_id * (batch_num - batch_id),
                        loss[0]))
            print("{}/{} epochs - cost time {:.0f}s - ETA {:.0f}s - ave loss: {:.4f} ...".format(
                str(epoch + 1).rjust(len(str(epochs))),
                epochs, time.time() - epoch_start, (time.time() - start) / (epoch + 1) * (epochs - epoch - 1), ave_loss))
        print("training complete, cost time {:.0f}s.".format(time.time() - start))

    def evaluate(self, vec_in, vec_out, batch_size=32, verbose_int=100):
        len_batch = 0
        ave_loss = 0.0
        for vec_in_, vec_out_, batch_id, batch_num in self.get_batch(vec_in, vec_out, batch_size=batch_size):
            len_ = len(vec_in_)
            vec_in_ = np.asarray(vec_in_, dtype='float32')
            vec_out_ = np.asarray(vec_out_, dtype='float32')
            feed_list = {'vec_in': vec_in_, 'vec_out': vec_out_}
            loss, = self.exe.run(self.test_pg, feed=feed_list, fetch_list=[self.loss])
            ave_loss = (loss[0] * len_ + ave_loss * len_batch) / (len_batch + len_)
            len_batch += len_
            if batch_id % verbose_int == 0:
                print("Eval - {}/{} batches - loss: {:.4f} ...".format(
                    str(batch_id).rjust(len(str(batch_num))), batch_num, loss[0]))
        print("Eval done - ave loss: {:.4f} ...".format(ave_loss))

    def trans(self, vec_in, batch_size=32):
        vec_out = []
        for vec_in_, batch_id, batch_num in self.get_batch(vec_in, batch_size=batch_size):
            len_ = len(vec_in_)
            vec_in_ = np.asarray(vec_in_, dtype='float32')
            vec_out_ = np.ones((len_, self.output_dim), dtype='float32')
            feed_list = {'vec_in': vec_in_, 'vec_out': vec_out_}
            pred = self.exe.run(self.test_pg, feed=feed_list, fetch_list=[self.pred])
            vec_out += pred
        return vec_out


if __name__ == '__main__':
    from vector_load import vec_load
    from lookup import EnVectorizer

    file_name = "glove.6B.50d.txt"
    file_path = "D:/OneDrive/WORK/word_vector"
    dct = vec_load(file_name, file_path, 10000)

    words = [w for w in dct.keys()]
    words = words[:10000]
    len_words = len(words)
    train_size = int(2/3 * len_words)
    train_vec_in = [dct[w] for w in words[:train_size]]
    test_vec_in = [dct[w] for w in words[train_size:]]

    ev = EnVectorizer(fast_mode=True, need_pro=False)
    ev.get_path(filename='glove.6B.300d.txt', path='D:/OneDrive/WORK/word_vector', pre_path='cache')
    train_vec_out = ev.lookup(words[:train_size], skip=False)
    test_vec_out = ev.lookup(words[train_size:], skip=False)

    vectrans = VecLinearTrans(input_dim=50, output_dim=300)
    vectrans.train(train_vec_in, train_vec_out, epochs=5, batch_size=32, verbose_int=10)
    vectrans.evaluate(test_vec_in, test_vec_out, batch_size=32, verbose_int=10)
