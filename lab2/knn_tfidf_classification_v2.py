from numpy import *
import numpy as np
import math

def KNN_classfication(): 
    f1 = open("classification_dataset/train_set.csv").readlines()[1:]
    vocab = {} #存储单词,第一位是单词序号，第二位是idf值
    labels = [] #存储标签 

    def get_tfidf_Data(): 
        dataSet = [] #store TFIDF matrix
        article_num = len(f1) #训练文本总数
        for line in f1:
            line = line.replace(',',' ').split()
            words = sorted(set(line[:-1]),key=line[:-1].index)
            labels.append([line[-1],0])
            for w in words:
                if w not in vocab.keys():
                    vocab[w] = [len(vocab),1] 
                else:
                    vocab[w][1] += 1  

        for w in vocab.keys():
            vocab[w][1] = math.log(float(article_num/vocab[w][1]), 2) #可用于之后的newdata

        for line in f1:
            seq = np.zeros(len(vocab))
            words = line.replace(',',' ').split()[:-1]
            len_words = len(words)
            for w in words:
                seq[vocab[w][0]] += 1  #用于TF矩阵
            words = sorted(set(words),key=words.index)
            for w in words:
                seq[vocab[w][0]] = seq[vocab[w][0]] / len_words * vocab[w][1] #TFIDF矩阵
            #Standard score            
            mean = sum(seq) / len(seq)
            std = np.std(seq,dtype='float64')
            seq = (seq - mean)/std

            dataSet.append(seq)
        return dataSet

    def calDist(x,y):
        x = np.array(x)
        y = np.array(y)
        dist = np.sqrt(np.sum(np.square(x - y))) #欧式距离
        #dist = dot(x,y)/(linalg.norm(x)*linalg.norm(y)) #余弦夹角
        return dist

    def knn_compare(dataSet, newdata, labels, k): #newdata指新的文本
        for i in range(len(dataSet)):
            dist = calDist(dataSet[i], newdata)
            labels[i][1] = dist #记录在对应的标签后面
        labels = sorted(labels,key=lambda x:x[1]) #key的效率比cmp高,递增排序(距离越短越接近)
        count = {}
        for i in range(k):
            key = labels[i][0]
            if key in count.keys():
                #count[key] += 1#直接加1
                count[key] += 1/labels[i][1]#根据距离长短来加
                #count[key] += 1/(1-labels[i][1])
                #count[key] += 1/log(labels[i][1]+1)
            else:
                #count[key] = 1
                count[key] = 1/labels[i][1]
        sortDict = ()
        sortDict = sorted(count.items(),key=lambda e:e[1], reverse=True)#e[1]指按值排序,reverse默认为False(升序排列))
        #sortDict是一个由元组组成的列表
        return sortDict[0][0]

    dataSet = get_tfidf_Data()
    #dataSet = get_tf_Data()
    f2 = open("classification_dataset/validation_set.csv").readlines()[1:]
    labels_valid = []
    labels_predict = []
    k = 13
    for line in f2:
        seq = np.zeros(len(vocab))
        line = line.replace(',',' ').split()
        words = line[:-1]
        len_words = len(words)
        for w in words:
            if w in vocab.keys():
                seq[vocab[w][0]] += 1
        words = sorted(set(words),key=words.index)
        for w in words:
            if w in vocab.keys():
                seq[vocab[w][0]] = seq[vocab[w][0]]/len_words * vocab[w][1] 

        mean = sum(seq) / len(seq)
        std = np.std(seq)
        seq = (seq-mean)/std

        labels_valid.append(line[-1])
        labels_predict.append(knn_compare(dataSet,seq,labels,k))
  
    right = 0
    for i in range(len(labels_valid)):
        if(labels_valid[i] == labels_predict[i]):
            right += 1
    accuracy = right / len(labels_valid)
    print(accuracy)

if __name__ == '__main__':
    import pdb
    #pdb.set_trace()
    np.seterr(divide='ignore', invalid='ignore')
    KNN_classfication()
    








