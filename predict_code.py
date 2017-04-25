# coding = utf-8

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from load_data import load_train_data, load_predict_data
from sklearn.datasets.base import Bunch
from sklearn.svm import SVC

import jieba

def jieba_tokenizer(x):
    return jieba.cut(x)

train_data = load_train_data(r"cuhk.csv")

def predict(n, x_test, y_test):
    #print(train_data)
    x_train, _, y_train, _ = train_test_split(train_data['data'], train_data['target'][n], test_size=0.5)
    #print(y_train)
    words_tfidf_vec = TfidfVectorizer(binary=False, tokenizer=jieba_tokenizer)
    X_train = words_tfidf_vec.fit_transform(x_train)
    print(train_data['types'][n])
    clf = SVC().fit(X_train, y_train)

    # 测试样本数据调用的是transform接口

    #print(x_test)
    X_test = words_tfidf_vec.transform(x_test)
    # 进行预测
    #print(x_test[:20])
    #print(clf.score(X_test, y_test))
    pred = clf.predict(X_test)
    #print(pred)
    # for label in pred:
    #     print(u'predict label: %s ' % label)
    return pred

if __name__=="__main__":

    #testing_data = load_train_data(r"cuhk.csv")
    #for i in range(10):
        #_, x_test, _, y_test = train_test_split(testing_data['data'], testing_data['target'][i], test_size=0.0002)
    #x_test = testing_data['data'][:10]
    x_test = load_predict_data()
    predict_result = [list() for i in range(len(x_test))]

    for i in range(10):

        result = predict(i, x_test, [])
        tmp = list()
        for j in range(len(result)):
            predict_result[j].append(result[j])

    print(x_test[:10])
    print(predict_result[:10])
    with open("answer1.csv", "w", encoding="utf-8") as f:
        for k, res in enumerate(predict_result):
            print(k)
            for i in range(9):
                f.write(str(res[i]))
                f.write(",")
            f.write(str(res[9]))
            f.write("\n")
