# Code from https://github.com/salesforce/awd-lstm-lm
import os
import torch as t

import collections


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = collections.Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:  #如果当前词没有在word2idx这个字典中，就将这个词放进word2idx这个列表中，word2idx这个列表保证了词不重复
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1 #给每个词分配一个id

        token_id = self.word2idx[word]  #对相同的词进行计数
        self.counter[token_id] += 1
        self.total += 1

        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary() #自定义的字典类
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.num_tokens = len(self.dictionary)

    def tokenize(self, path):
        """Tokenizes a text file.
        将词加入Corpus中自定义的字典属性
        然后将每个词所对应的字典key保存到一个ids中
        params：path：文本文件所在目录
        return：词对应的序号LongTensor
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>'] #以空格进行分析然后每行最后加一个<eos>
                tokens += len(words)             #对词进行计数 929859(train)
                for word in words:
                    self.dictionary.add_word(word)  #将词加入Corpus类的Dictionary属性中，这个字典是自定义的

        # Tokenize file content
        with open(path, 'r') as f:
            ids = t.LongTensor(tokens)#创建一个tokens行，1列的一个LongTensor（也就是这个Tensor里面的数据是64bit的）
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]  #依次将每个词所对应的序号放入到ids这个Tensor中
                    token += 1

        return ids  #ids([929859])
