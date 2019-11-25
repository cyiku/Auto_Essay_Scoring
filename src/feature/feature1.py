# encoding=utf-8

import numpy as np

from src.util.tfidf import TFIDF, TFTF
from src.util.util import matrix_cosine_similarity, pos_tagging, gram

"""
输入分为两种： sequence or  tokens
"""


def word_vector_similarity_train(train_data, score_list, TFIDF_class=None):
    """ input: tokens (已经tokenizer的)"""

    sample_num = len(score_list)

    train_data = TFIDF.process_data(train_data)
    if not TFIDF_class:
        TFIDF_class = TFIDF(train_data)

    tfidf = TFIDF_class.get_train_tfidf()
    cosine = matrix_cosine_similarity(tfidf)
    score_list = np.array(score_list)

    attn = score_list * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (sample_num - 1)
    return result.reshape(sample_num, 1)


def word_vector_similarity_test(test_data, train_score_list, TFIDF_class: TFIDF):
    """ input: tokens (已经tokenizer的)"""

    assert TFIDF_class, u"TFIDF的类必须使用同样题目的训练集合的TFIDF"
    sample_num = len(train_score_list)

    test_data = TFIDF.process_data(test_data)

    tfidf = TFIDF_class.get_test_tfidf(test_data)
    cosine = matrix_cosine_similarity(tfidf)
    score_list = np.array(train_score_list)

    attn = score_list * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (sample_num - 1)
    return result.reshape(sample_num, 1)


def pos_bigram_train(train_data, TFTF_class=None):
    """ input: tokens"""

    # 1. 词性标注
    tagged_data = pos_tagging(train_data)

    # 2. 组成2-gram
    gramed_data = gram(tagged_data, 2)

    join_data = [' '.join(d) for d in gramed_data]
    if not TFTF_class:
        TFTF_class = TFTF(join_data)

    train_tfTF = TFTF_class.get_train_tfTF()

    return train_tfTF


def pos_bigram_test(test_data, TFTF_class: TFTF):
    """ input: tokens (已经tokenizer的)"""

    assert TFTF_class is not None, u"测试阶段，TFTF类不能为None"
    # 1. 词性标注
    tagged_data = pos_tagging(test_data)

    # 2. 组成2-gram
    gramed_data = gram(tagged_data, 2)

    # 3. 计算tfTF
    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = TFTF_class.get_test_tfTF(join_data)

    return test_tfTF


def mean_clause(data):
    """
    train test使用
    """
    assert data is not None, u"data不能为none"
