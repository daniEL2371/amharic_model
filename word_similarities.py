import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[33]:



def get_similarity_on(word1, word2, startIndex):
    sim = 0
    for i in range(len(word2)):
        if word1[i + startIndex] == word2[i]:
            sim += 1
    return sim


def get_similarity(word1, word2):
    max_sim = 0
    if len(word1) == 0 or len(word2) == 0:
        return 0
    if len(word1) < len(word2):
        word2, word1 = word1, word2
    r = len(word1) - len(word2)
    for j in range(r + 1):
        sim = get_similarity_on(word1, word2, j)
        if sim > max_sim:
            max_sim = sim
    return max_sim / len(word1)


def add_to_sim_list(sim_dict, w1, w2, sim):
    if w1 in sim_dict:
        sim_dict[w1].append((w2, sim))
    else:
        sim_dict[w1] = [(w2, sim)]


# In[38]:


filename = "data/news.txt"
vocabulary = open(filename, encoding='utf-8').read().split(' ')[:1000]
print("Unpure Vocab Size: ", len(set(vocabulary)))
vocabulary_size = 0

def contains_digit(word):
    for i in range(10):
        if str(i) in word:
            return True
    return False


def build_dataset(words):
    dictionary = dict()
    for word in words:
        if contains_digit(word):
            word = 'UNK'
        if word not in dictionary:
            dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary

word2int, int2word = build_dataset(vocabulary)
del vocabulary
VOCAB_SIZE = len(word2int)
print("Vocab Size: ", VOCAB_SIZE)


# In[39]: 


sim_dict = {}
for i in range(VOCAB_SIZE):
    for j in range(VOCAB_SIZE):
        w1, w2 = int2word[i], int2word[j]
        sim = get_similarity(w1, w2)
        if sim > 0.3:
            add_to_sim_list(sim_dict, w1, w2, sim)


# In[41]:


roots = {}
k = 0
for key in sim_dict:
    for l in sim_dict[key]:
        word, weight = l
        if word not in roots:
            roots[word] = (k, weight)
        else:
            rk, rw = roots[word]
            if rw < weight:
                roots[word] = (k, weight)
    k += 1

g = nx.Graph()
graph = []
for word1 in sim_dict:
    for word2, weight in sim_dict[word1]:
        graph.append((word1, word2, weight))

g.add_weighted_edges_from(graph)
plt.rc('font', family='Ebrima')
# plt.subplot(111)
nx.draw(g, with_labels=True, font_family="Ebrima")
plt.show()

