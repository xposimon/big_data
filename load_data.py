# coding = utf-8
import codecs
import numpy as np
from sklearn.datasets.base import Bunch

def trans(m):
    return list(map(list,zip(*m)))

def load_train_data(filename = "cuhk.csv"):
    with codecs.open(filename, "r", encoding="utf-8") as f:
        raw_data = f.readlines()
        types = raw_data[0].strip().split(",")

    # with open("test_data.data", "w", encoding="utf-8") as f:
    #     for line in raw_data[:30]:
    #         f.write(line.strip() + '\n')

    data = dict()
    for name in types:
        data[name] = dict()

    clean_data = []

    for line in raw_data:
        now_line = line.split(',')
        if len(now_line) != 12 or (now_line[2] != '1' and now_line[2] != '-1' and now_line[2] != '0'):
            continue
        now_line[11] = now_line[11].strip()
        for i in range(2, len(now_line)):
            now_line[i] = eval(now_line[i])
        clean_data.append(now_line[1:])

    #print(clean_data)

    test_data = trans(clean_data)

    return {'data':test_data[0], 'target':test_data[1:], 'types':types[2:]}


def load_predict_data(filename = "raw_data_test.csv"):
    with codecs.open(filename, "r", encoding="utf-8") as f:
        test_data = f.readlines()[::2]
    for i in range(len(test_data)):
        test_data[i] = test_data[i].strip()
    return test_data

if __name__ == "__main__":
    load_train_data()
    print(np.zeros(3))
