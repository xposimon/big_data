# coding = utf-8

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from load_data import load_train_data, load_predict_data
from sklearn.datasets.base import Bunch
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import jieba

def jieba_tokenizer(x):
    return jieba.cut(x)

train_data = load_train_data(r"cuhk.csv")

def GBC(X_train, y_train, X_test):
    clf = GradientBoostingClassifier(n_estimators=1000, max_depth=14)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test.toarray())  # for GBC
    return pred

def linearSVC(X_train, y_train, X_test):
    clf = LinearSVC().fit(X_train, y_train)
    pred = clf.predict(X_test)  # Linear SVC
    return pred

def decisionTree(X_train, y_train, X_test):
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # summarize the fit of the model
    return pred

def predict(n, x_test, y_test):
    #print(train_data)
    x_train, _, y_train, _ = train_test_split(train_data['data'], train_data['target'][n], test_size=0.98)
    #print(y_train)
    words_tfidf_vec = TfidfVectorizer(binary=False, tokenizer=jieba_tokenizer)
    X_train = words_tfidf_vec.fit_transform(x_train)
    print(train_data['types'][n])
    print(X_train.shape[0])

    X_test = words_tfidf_vec.transform(x_test)

    #print(clf.score(X_test, y_test))
    #print(pred)


    return GBC(X_train, y_train, X_test)

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
