# encoding=utf-8

import unittest

from src.util.tfidf import TFIDF, TFTF
from src.util.util import word_stemming_remove_stop, tokenizer, pos_tagging, ngram, constituenty_tree


class TestUtil(unittest.TestCase):

    def test_get_tfidf(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]
        corpus = TFIDF.process_data(corpus)
        print(corpus)

        tfidf_class = TFIDF(corpus)

        tfidf = tfidf_class.get_train_tfidf()
        word = tfidf_class.get_word()
        idf = tfidf_class.get_idf()
        tf = tfidf_class.get_train_tf()
        print(tfidf)

        # print('---------------')
        # print(word)
        # print('----------------')
        # print(idf)
        # print('-----------------')
        # print(tf)

        test = [
            'first first document',
            'This is the first document.',
            'second Is one'
        ]

        test = TFIDF.process_data(test)
        print(test)

        tfidf = tfidf_class.get_test_tfidf(test)
        # word = tfidf_class.get_word()
        # idf = tfidf_class.get_idf()
        print(tfidf)
        print('---------------')
        # print(word)
        # print('----------------')
        # print(idf)

    def test_word_stemming_remove_stop(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?'
        ]

        corpus = tokenizer(corpus)
        print(corpus)

        data = word_stemming_remove_stop(corpus)
        print(data)

    def test_pos_tag(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?'
        ]

        corpus = tokenizer(corpus)
        # print(corpus)

        data = pos_tagging(corpus)
        return data

    def test_bigram(self):
        data = self.test_pos_tag()

        data = ngram(data)

        return data

    def test_get_tftf(self):
        data = self.test_pos_tag()

        data = ngram(data)
        data = [' '.join(d) for d in data]
        print(data)

        TFTF_class = TFTF(data)
        print(TFTF_class.get_word())
        print(TFTF_class.get_train_tfTF())
        print(TFTF_class.get_train_TF())

        test = [
            'first first document',
            'This is the first document.',
            'second Is one'
        ]
        test = tokenizer(test)
        # rint(test)

        test = pos_tagging(test)
        test = ngram(test)
        test = [' '.join(d) for d in test]
        print(test)

        print(TFTF_class.get_test_tfTF(test))
        print(TFTF_class.get_train_TF())

    def test_constituenty_tree(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?'
        ]

        result = constituenty_tree(corpus)

        return result
