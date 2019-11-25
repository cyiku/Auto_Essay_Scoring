# encoding=utf-8
import argparse

from src.data import Dataset
from src.feature.feature import Feature


def train():
    # 1. 加载数据集
    print("start loading data_set")
    train_dataset: Dataset = Dataset.load("../data/essay_data/train.pickle")
    dev_dataset: Dataset = Dataset.load("../data/essay_data/dev.pickle")
    test_dataset: Dataset = Dataset.load("../data/essay_data/test.pickle")
    print("end loading data_set")

    # 2. 计算特征
    essay_set_num = len(train_dataset.data)
    for set_id in range(1, essay_set_num + 1):
        train_data = train_dataset[str(set_id)]
        dev_data = dev_dataset[str(set_id)]
        test_data = test_dataset[str(set_id)]
        train_every_set(train_data, dev_data, test_data, set_id)


def train_every_set(train_data, dev_data, test_data, set_id):
    """ 训练每一个set """
    feature_class = Feature()
    print("start compute the feature for essay set ", set_id)
    feature = feature_class.get_feature(train_data)
    print("end compute the feature for essay set ", set_id)
    # 3. 构建模型，训练

    # 4. 保存模型，输出结果


def test():
    # 1. 先train
    # 2. 计算feature
    # 3. 测试，输出结果
    pass


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--run", type=str, default='train', help='train or test', choices=['train', 'test'])
    args = parse.parse_args()

    run = args.run

    if run == 'train':
        train()
    elif run == 'test':
        test()
    else:
        assert False, u"纳尼，居然还有这个选择能进来"
