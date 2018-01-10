import numpy as np
import random

class Data():
    def __init__(self, mode):

        self.mode = mode
        self.char_list = None 
        self.char_dict = None 
        self.star_train = None 
        self.text_train = None
        self.len_train = None
        self.star_test = None 
        self.text_test = None
        self.len_test = None

    def load(self):
        print("Loading data...")
        if self.mode==3:
            self.char_list = np.load("../dataset/id2word.npy")
            self.char_dict = np.asarray(np.load("../dataset/id2glove.npy"))
            self.star_train = np.load("../dataset/train_star_w2v.npy")
            self.text_train = np.load("../dataset/train_text_w2v.npy")
            self.len_train = np.load("../dataset/train_len_w2v.npy")
            self.star_test = np.load("../dataset/test_star_w2v.npy")
            self.text_test = np.load("../dataset/test_text_w2v.npy")
            self.len_test = np.load("../dataset/test_len_w2v.npy")
        elif self.mode==1:
            self.char_list = np.load("../dataset/char_list_full.npy")
            self.char_dict = np.load("../dataset/char_dict_full.npy").item()
            self.star_train = np.load("../dataset/train_star_full.npy")
            self.text_train = np.load("../dataset/train_text_full.npy")
            self.len_train = np.load("../dataset/train_len_full.npy") # meanless
            self.star_test = np.load("../dataset/test_star_full.npy")
            self.text_test = np.load("../dataset/test_text_full.npy")
            self.len_test = np.load("../dataset/test_len_full.npy") # meanless
        else:
            self.char_list = np.load("../dataset/char_list.npy")
            self.char_dict = np.load("../dataset/char_dict.npy").item()
            self.star_train = np.load("../dataset/train_star.npy")
            self.text_train = np.load("../dataset/train_text.npy")
            self.len_train = np.load("../dataset/train_len.npy")
            self.star_test = np.load("../dataset/test_star.npy")
            self.text_test = np.load("../dataset/test_text.npy")
            self.len_test = np.load("../dataset/test_len.npy")
        print("Train/Test: {:d}/{:d}".format(len(self.star_train), len(self.star_test)))
        print("==================================================================================")

        return len(self.star_train), len(self.star_test)

    def train_batch_iter(self, batch_size):
        data = list(zip(self.star_train, self.text_train, self.len_train))
        return self.batch_iter(data, batch_size)

    def test_batch_iter(self, batch_size):
        data = list(zip(self.star_test, self.text_test, self.len_test))
        return self.batch_iter(data, batch_size)

    def batch_iter(self, data, batch_size):
        data_size = len(data)
        num_batches = int(len(data)/batch_size)
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]
