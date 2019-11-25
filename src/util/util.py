# encoding=utf-8

"""
计算一些统计信息
"""
import nltk
from nltk.corpus import stopwords

stoplist = stopwords.words('english')

from stanfordcorenlp import StanfordCoreNLP

from src.config import STANFORDCORENLP_PATH

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


def remove_stop_word(data):
    """ remove stop word """

    result = [[token for token in sample if token not in stoplist] for sample in data]
    return result


def word_stemming(data):
    """ word stemming """

    stemmer = PorterStemmer()

    result = [[stemmer.stem(token) for token in sample] for sample in data]
    return result


def word_stemming_remove_stop(data):
    """ word stemming and remove stop word """

    stemmer = PorterStemmer()
    result = [[stemmer.stem(token) for token in sample if token not in stoplist] for sample in data]
    return result


def tokenizer(data):
    result = [word_tokenize(d) for d in data]
    return result


def vector_cosine_similarity(vector1, vector2):
    """ compute the cosine similarity """

    assert vector1 and vector2, u"vector_cosine_similarity中, vector1和vector2不能为none"

    result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

    return result


def matrix_cosine_similarity(matrix1, matrix2=None):
    """ 为矩阵的元素两两计算余弦相似度 """

    assert matrix1 is not None, u"matrix_cosine_similarity中, matrix1不能为none"

    if matrix2 is None:
        matrix2 = matrix1

    cosine = cosine_similarity(matrix1)
    return cosine


def clause_split(data):
    """ clause detect """
    pass


def pos_tagging(train_data):
    """ 词性标注 """
    result = []
    for sample in train_data:
        pos_tag = nltk.pos_tag(sample)
        result.append([_tag[1] for _tag in pos_tag])
    return result


def ngram(train_data, n=2, join_char='_'):
    """ n-gram """
    result = []
    for sample in train_data:
        bi_sample = nltk.ngrams(sample, n)
        result.append([join_char.join(s) for s in bi_sample])

    return result


def constituenty_tree(train_data):
    """ 计算mean clause使用
     input: sentences"""

    result = []
    # print(STANFORDCORENLP_PATH)
    nlp = StanfordCoreNLP(STANFORDCORENLP_PATH)
    for sentence in train_data:
        result.append(nlp.parse(sentence))

    nlp.close()
    return result

