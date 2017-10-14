import numpy as np
import math
import pdb

def NB_classification():
    f1=open("classification_dataset/train_set.csv").readlines()[1:]
    article_total = len(f1) #总文本数
    labels_info = {} #存储标签总文本数,对应的总单词数,对应总不重复单词数(不同文本中有相同单词),以及不同单词出现次数

    for line in f1:
        line = line.replace(',',' ').split()
        label = line[-1]
        if label not in labels_info:
            labels_info[label] = [1,len(line[:-1]),0,{}]
        else:
            labels_info[label][0] += 1
            labels_info[label][1] += len(line[:-1])
        for w in line[:-1]:
            if w not in labels_info[label][3].keys():
                labels_info[label][3][w] = 1
            else:
                labels_info[label][3][w] += 1
    #for label in labels_info:
    #    labels_info[label][2] = len(labels_info[label][3])
    count = {}
    for line in f1:
        line = line.replace(',',' ').split()
        words = line[:-1]
        for w in words:
            if w not in count.keys():
                count[w] = 0
    cnt = len(count)
               
    f2 = open("classification_dataset/validation_set.csv").readlines()[1:]
    label_predict = []
    label_valid = []
    for line in f2:
        line = line.replace(',',' ').split()
        prob_dif = {}
        label_valid.append(line[-1])
        for label in labels_info:
            prob_dif[label] = labels_info[label][0]/article_total #p(ei)
            words = sorted(set(line[:-1]),key=line[:-1].index)
            for w in words:
                if w in labels_info[label][3]:
                    #prob_w = (labels_info[label][3][w]+0.001) / (labels_info[label][1]+labels_info[label][2])
                    #prob_w = math.log((labels_info[label][3][w]+0.001) / (labels_info[label][1]+cnt))
                    prob_w = (labels_info[label][3][w]+1) / (labels_info[label][1]+cnt)
                else:
                    #prob_w = 0.001/(labels_info[label][1]+labels_info[label][2])
                    #prob_w = math.log((labels_info[label][3][w]+0.001) / (labels_info[label][1]+cnt))
                    prob_w = 1/(labels_info[label][1]+cnt)
                prob_dif[label] *= prob_w
                
            prob_dif[label] = math.log(prob_dif[label])
        sortDict = ()
        sortDict = sorted(prob_dif.items(),key=lambda e:e[1],reverse=True)#e[1]指按值排序
        #print(sortDict)
        label_predict.append(sortDict[0][0])

    right = 0
    for i in range(len(label_valid)):
        if(label_valid[i] == label_predict[i]):
            right += 1
    accuracy = right / len(label_valid)
    print(accuracy)

if __name__ == '__main__':
    #pdb.set_trace()
    NB_classification()













