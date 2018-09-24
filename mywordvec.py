import numpy as np 
import tensorflow as tf 

corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = corpus_raw.lower()

words  = []
for word in corpus_raw.split():
    if word != ',':
        words.append(word)

words = set(words)

word2int = {}
int2word = {}

vocab_size = len(words)

for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

