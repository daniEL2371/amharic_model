import numpy as np

class DataGen:

    def __init__(self):
        self.word2int = None
        self.int2word = None
        self.vocab = None

    def to_1_hot(self, index, vocab_size):
        temp = np.zeros(vocab_size)
        temp[index] = 1
        return temp

    def tokenize(self, corpus):
        corpus_raw = open(corpus, encoding='utf-8').read()
        lines = corpus_raw.split('\n')
        lines = lines[:10]
        words = []
        for line in lines:
            for word in line.split():
                words.append(word)

        words = set(words)
        self.word2int = {}
        self.int2word = {}

        vocab_size = len(words)
        for i, word in enumerate(words):
            self.word2int[word] = i
            self.int2word[i] = word
        self.vocab = words
        return lines, self.vocab, self.word2int, self.int2word

    def vectorize(self, lines, window_size):
        sentenses = []
        for line in lines:
            sens = line.split()
            sentenses.append(sens)

        data = []
        for sentense in sentenses:
            for word_index, word in enumerate(sentense):
                for nb_word in sentense[max(word_index - window_size, 0): min(word_index + window_size, len(sentense)) + 1]:
                    if nb_word != word:
                        data.append([word, nb_word])

        return data

    def get_data(self, data):
        x_train = []
        y_train = []
        vocab_size = len(self.vocab)
        for data_word in data:
            x_train.append(self.to_1_hot(self.word2int[data_word[0]], vocab_size))
            y_train.append(self.to_1_hot(self.word2int[data_word[1]], vocab_size))

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        return x_train, y_train

    def gen_data(self, corpus, batch_size=100, n_batchs=-1, window_size=2, embedd_size=64):
        lines, vocab, word2int, int2word = self.tokenize(corpus)
        data = self.vectorize(lines, window_size)
        x_data, y_data = self.get_data(data)
        n_data = x_data.shape[0]
        batch = 0
        print("Starting Generating data")
        while True:
            if n_batchs == batch or (batch * batch_size) >= n_data:
                batch = 0
            current_i = batch_size * batch
            upto_i = batch_size * (batch + 1)
            x = x_data[current_i:upto_i]
            y = y_data[current_i:upto_i]
            yield x, y
            batch += 1

