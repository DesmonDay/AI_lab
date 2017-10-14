def KNN_regression():
    f1 = open("regression_dataset/train_set.csv").readlines()[1:]
    vocab = {} #存储单词,第一位是单词序号，第二位是idf值
    labels = []

    def getData():
        dataSet = [] #store TFIDF matrix
        article_num = len(f1) #训练文本总数
        for line in f1:
            line = line.replace(',',' ').split()
            words = sorted(set(line[:-6]),key=line[:-6].index)
            labels.append([float(line[-6]),float(line[-5]),float(line[-4]),float(line[-3]),float(line[-2]),float(line[-1]),0])
            for w in words:
                if w not in vocab.keys():
                    vocab[w] = [len(vocab),1]
                else:
                    vocab[w][1] += 1

        for w in vocab.keys():
            vocab[w][1] = math.log(float(article_num/vocab[w][1]), 2)
            
        for line in f1:
            seq = np.zeros(len(vocab))
            words = line.replace(',',' ').split()[:-6]
            len_words = len(words)
            for w in words:
                seq[vocab[w][0]] += 1 #用于TF矩阵
            words = sorted(set(words),key=words.index)
            for w in words:
                seq[vocab[w][0]] = seq[vocab[w][0]]/len_words*vocab[w][1]
            #Standard score
            '''
            mean = sum(seq) / len(seq)
            std = np.std(seq)
            seq = (seq-mean)/std
            '''          
            dataSet.append(seq)
        return dataSet

    def calDist(x,y):
        x = np.array(x)
        y = np.array(y)
        #dist = np.sqrt(np.sum(np.square(x - y))) #欧式距离
        dist = dot(x,y)/(linalg.norm(x)*linalg.norm(y)) #余弦夹角
        return dist

    def knn_compare(dataSet, newdata, labels, k):
        for i in range(len(dataSet)):
            dist = calDist(dataSet[i], newdata)
            dist = 1-dist
            labels[i][6] = dist
        labels = sorted(labels, key=lambda x:x[6]) #升序
        #labels = sorted(labels, key=lambda x:x[6], reverse=True)#降序，用于余弦夹角
        probability = [0,0,0,0,0,0]
        for i in range(k): #
            probability[0] += labels[i][0]/labels[i][6] #anger
            probability[1] += labels[i][1]/labels[i][6] #disgust
            probability[2] += labels[i][2]/labels[i][6] #fear
            probability[3] += labels[i][3]/labels[i][6] #joy
            probability[4] += labels[i][4]/labels[i][6] #sad
            probability[5] += labels[i][5]/labels[i][6] #surprise
        return probability

    dataSet = getData()
    f2 = open("regression_dataset/validation_set.csv").readlines()[1:]
    k = 9
    f3 = open("regression_dataset/knn_result_set.csv", "w")
    for line in f2:
        seq = np.zeros(len(vocab))
        line = line.replace(',',' ').split()
        words = line[:-6]
        len_words = len(words)

        for w in words:
            if w in vocab.keys():
                seq[vocab[w][0]] += 1
        words = sorted(set(words),key=words.index)
        for w in words:
            if w in vocab.keys():
                seq[vocab[w][0]] = seq[vocab[w][0]]/len_words * vocab[w][1] 
        '''
        mean = sum(seq) / len(seq)
        std = np.std(seq)
        seq = (seq-mean)/std
        '''

        probability = knn_compare(dataSet, seq, labels, k)
        s = sum(probability)
        print(s)
        for i in range(len(probability)):
            probability[i] = probability[i] / s
        f3.write(str(probability[0])+','+str(probability[1])+','+str(probability[2])
         +','+str(probability[3])+','+str(probability[4])+','+str(probability[5])+'\n')

if __name__ == '__main__':
    from numpy import *
    import numpy as np
    import math
    #import pdb
    #pdb.set_trace()
    np.seterr(divide='ignore', invalid='ignore')
    KNN_regression()













