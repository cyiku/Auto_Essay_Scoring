# encoding=utf-8
from src.data import Dataset

#
from src.feature.feature1 import word_vector_similarity_train, pos_bigram_train, word_vector_similarity_test


class Feature:
    """
    我也不知道定义类还是函数！！！
    """

    def __init__(self):
        pass

    def get_feature(self, train_data_set: Dataset):
        """ 不用区分训练测试的, 可以写在这类"""

        feature = self.get_train_feature(train_data_set)
        return feature

    def get_train_feature(self, train_data_set: Dataset):
        set_id = 1
        data, score_list = train_data_set.get_data_list(set_id)
        wv_similarity = word_vector_similarity_train(data, score_list)
        print(wv_similarity)

        pos_bigram = pos_bigram_train(data)
        print(pos_bigram)

        # TODO 拼接
        return

    def get_test_feature(self):
        # set_id = 1
        # data, score_list = test_data_set.get_data_list(set_id)
        #
        # wv_similarity = word_vector_similarity_test(data, score_list)
        # print(wv_similarity)
        #
        # pos_bigram = pos_bigram_train(data)
        # print(pos_bigram)
        pass
