from numpy import *
import pdb
import numpy as np
import math

def NB_regression():
    f1 = open("regression_dataset/train_set.csv").readlines()[1:]
    labels_info = []
    nonrepetitive_words = 0
    for line in f1:
        words = line.replace(',',' ').split()[:-6]
        nonrepetitive_words += len(words)

    for line in f1:
        words = line.replace(',',' ').split()
        vocab = {} #存储TF
        for w in words[:-6]:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
        for w in vocab:
            vocab[w] = (vocab[w] + 1) / (len(words[:-6]) + nonrepetitive_words) #平滑后的TF
            #vocab[w] = (vocab[w] + 0.001) / (len(words[:-6]) + nonrepetitive_words) #平滑后的TF
            #vocab[w] = (vocab[w] + 0.001) / len(words[:-6])

        emotion = []
        for w in words[-6:]:
            emotion.append(float(w))
        file = []
        file.append(vocab);file.append(emotion);file.append(len(words[:-6]));
        labels_info.append(file)

    f2 = open("regression_dataset/validation_set.csv").readlines()[1:]
    f3 = open("regression_dataset/nb_result_set.csv","w")
    for line in f2:
        words = line.replace(',',' ').split()[:-6]
        probability = []
        for i in range(len(labels_info[0][1])): #算6个标签
            prob_i = 0
            for j in range(len(labels_info)):
                pro = labels_info[j][1][i] #给定标签概率
                for w in words:
                    if w in labels_info[j][0]:
                        pro *= labels_info[j][0][w]
                    else:
                        pro *= 1/(labels_info[j][2]+nonrepetitive_words)
                        #pro *= 0.001/(labels_info[j][2]+nonrepetitive_words)
                        #pro *= 0.001/labels_info[j][2]
                prob_i += pro
            probability.append(prob_i)

        #print(probability)
        s = sum(probability)
        for p in range(len(probability)):
            probability[p] = probability[p] / s
        f3.write(str(probability[0])+','+str(probability[1])+','+str(probability[2])
         +','+str(probability[3])+','+str(probability[4])+','+str(probability[5])+'\n')


if __name__ == "__main__":
    #pdb.set_trace()
    NB_regression()


    






