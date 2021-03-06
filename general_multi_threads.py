#coding=utf-8

import nltk, form_data, jieba, logic_dic
import jieba.posseg as pseg
from synonym import similarity
from threading import Thread

class general_algorithm:
# A simple and general algorithm to classify
# The format of feature_set: (feature_dic, judgement)
# feature_dic e.g. {'noun':'洗发水', 'adj':'好用', 'verb':'是'}
########################################################################################################################

    def __init__(self):
    # Get the classified dic
        self.dictionary = form_data.dictionary
        self.adj = logic_dic.logic_list
        self.classes = ['price', 'fakeconcern', 'promotion', 'service', 'leakage', 'package',
                        'loyalty', 'smell', 'effect', 'logistics']
        self.dic_tags = ['n', 'a']
        self.classifier = {}

    def compare(self, word, word_set):
    # Judge whether a word is in the set, or a synonym of one of them
        if word in word_set:
            return True
        # Synonym , similarity bigger than 0.65 is considered to be the same
        for word_ in word_set:
            if similarity(word, word_) > 0.65:
                return True

        return False

    def get_train_feature(self, comment, now_class, judgement = None):
    # Parse a comment and get features

        word_set = self.dictionary[now_class]
        adj_set = self.adj

        if judgement:
            judgement = judgement.strip()
        feature_set = {}

        parsed_comment = jieba.cut(comment, cut_all=True)

        # Find features
        for word in parsed_comment:
            if self.compare(word, word_set):
                for _word, flag in pseg.cut(word):
                    if _word not in feature_set.keys():
                        feature_set[_word] = flag[0]

        for word in parsed_comment:
            if self.compare(word, adj_set):
                for _word, flag in pseg.cut(word):
                    if _word not in feature_set.keys():
                        feature_set[_word] = flag[0]
        #print(feature_set)
        return feature_set


    def train(self, train_data_path = 'train_data.data'):
    # Train data format:
    # 1,双十一抢购 抢了一个晚上4拿下 一百的优惠券没有用上,0,0,1,0,0,0,0,0,0,0
    # id,Text,Price,Fakeconcern,Promotion,Service,Leakage,Package,Loyalty,Smell,Effect,Logistics

        with open(train_data_path, encoding='utf-8') as train_data_file:
            train_data = train_data_file.readlines()

        for class_order in range(10):
            now_class = self.classes[class_order]
            train_set = []
            for line_data in train_data:
                now_line = line_data.split(',')
                if len(now_line) != 12 or (now_line[2] != '1' and now_line[2] != '-1'):
                    continue
                now_line[11] = now_line[11].strip()
                feature_set = self.get_train_feature(now_line[1], now_class, now_line[2+class_order])
                #print(now_line[2+class_order])
                train_set.append((feature_set, now_line[2+class_order]))

                    #print(now_line, 2+class_order)
            #print(train_set,"!!1")
            self.classifier[now_class] = nltk.NaiveBayesClassifier.train(train_set)

    def judge(self, comment):
        judgement = {}
        p = []
        for _class in self.classes:
            judgement[_class] = {}
            feature_set = self.get_train_feature(comment, _class)
            if len(feature_set) == 0:
                now_judgement = '0'
            else :
                now_judgement = self.classifier[_class].classify(feature_set)
            # print(_class + ":" + now_judgement)
            p.append(str(now_judgement))
        return p

def multi_test(lines, count):
    with open('test_result'+str(count)+'.csv', 'w', encoding="utf-8") as w:
        for line in lines:
            line = line.strip()
            if not line:
                r.write('\n')
                continue
            plist = test.judge(line)
            w.write(','.join(plist))
            w.write('\n')


if __name__ == '__main__':
    test = general_algorithm()
    test.train()
    count = 0
    with open('raw_data_test.csv', 'r', encoding='utf-8') as f:
        input = f.readlines()
        length = len(input)
        threads = length / 1000
        for i in range(threads):
            t = Thread(target=multi_test, args=(input[i*100:(i+1)*100], i,))
            t.start()
            t.join()

        with open("test_result.csv", "w", encoding="utf-8") as w:
            w.write()

        with open("test_result.csv", "a", encoding="utf-8") as w:
            for i in range(threads):
                with open("test_result"+str(i)+".csv", "r", encoding="utf-8") as r:
                    w.write(r.read())

