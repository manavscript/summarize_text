import re
import numpy as np
import numpy.linalg as la
import collections
import sys
import math


"""
Based on https://github.com/facebookarchive/NAMAS
Trying to replicate and test the extractive abstraction
NLP LSA Summarization
"""
np.set_printoptions(threshold=sys.maxsize)


class LSA_Summarization():
    rows = 0
    columns = 0
    matrix = None
    word_list = None
    sentence_list = None
    u = None
    sigma = None
    v_t = None
    s_k = None

    def __init__(self, filename):
        with open(filename, encoding="utf8") as f:
            raw_text = f.read()
        raw_text = raw_text.replace("\n", "")
        sentence_list = re.split(r' *[\.\?!][\'"\)\]]* *', raw_text)
        word_list = re.split(r"\s+", raw_text)
        self.columns = len(sentence_list)
        self.rows = len(word_list)
        self.matrix = np.zeros(shape=(self.rows, self.columns))
        self.word_list = word_list
        self.sentence_list = sentence_list
        for r, word in enumerate(word_list):
            for c, sentence in enumerate(sentence_list):
                words_in_sentence = re.split(r"\s+", sentence)
                count = collections.Counter(words_in_sentence)
                self.matrix[r][c] = count[word]
    
    def svd(self):
        self.u, self.sigma, self.v_t = la.svd(self.matrix)
        # print("\n", self.u, "\n", self.v_t, "\n", self.sigma)
        sigma_sqaured = np.square(self.sigma, self.sigma)

        self.s_k = []
        for column in self.v_t.T:
            s_kth = sum(s*v**2 for s, v in zip(sigma_sqaured, column))
            self.s_k.append(s_kth)
        # print("\n\n")
        # print(self.s_k)

    def choose_sentences(self, ratio, filename):
        num_sentences = math.ceil(ratio*len(self.sentence_list))
        arr = np.zeros(shape=(2, len(self.s_k)))
        for index, val in enumerate(self.s_k):
            arr[0][index] = index
            arr[1][index] = val
        arr = arr [ :, arr[1].argsort()]
        index_to_choose = []
        for i in range(num_sentences):
            index_to_choose.append(arr[0][[i]])
        index_to_choose = sorted(index_to_choose)
        chosen_list = []
        for i in range(num_sentences):
            print(int(index_to_choose[i][0]), i)
            chosen_list.append(self.sentence_list[int(index_to_choose[i][0])])
        with open(filename, "w") as f:
            for sentence in chosen_list:
                f.write(sentence)
                f.write(".")
                f.write("\n")
        return chosen_list




if __name__ == "__main__":
    b = LSA_Summarization("text.txt")
    b.svd()
    print(b.choose_sentences(0.4, "summarized.txt"))
