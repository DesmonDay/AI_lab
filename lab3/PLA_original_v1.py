import numpy as np

def judge(x, y, label): #不等于给定标签，则返回false，重新计算w
    predict = 0
    if sum(x*y) > 0:
        predict = 1
    elif sum(x*y) < 0:
        predict = -1
    if predict == label: 
        return 1,predict
    else: 
        return 0,predict

def PLA(iteration):
    f1 = open("train.csv").readlines()
    trainMax = []
    labels = []
    totals = 0
    for line in f1:
        totals += 1
        line = line.split(',')
        labels.append(float(line[-1]))
        temp = []
        temp.append(1)
        for w in line[:-1]:
            temp.append(float(w))
        trainMax.append(temp)

    trainMax = np.array(trainMax)
    #w = np.ones(len(trainMax[0]))#初始化权值，均为1
    w = np.zeros(len(trainMax[0]))

    for i in range(iteration):
        for j in range(totals):
            if judge(w, trainMax[j], labels[j])[0] == 1:
                continue
            else:
                w = w + labels[j]*trainMax[j]
                break
    return w

def Predict(w):    
    

    f2 = open("val.csv").readlines()
    valMax = []
    val_labels = []
    total = 0
    for line in f2:
        total += 1
        line = line.split(',')
        val_labels.append(float(line[-1]))
        temp = []
        temp.append(1)
        for word in line[:-1]:
            temp.append(float(word))
        valMax.append(temp)
    
    valMax = np.array(valMax)
    predict_label = []
    tp = fn = tn = fp = 0

    for j in range(total):
        #right += judge(w, valMax[j], val_labels[j])[0]
        predict_label.append(judge(w, valMax[j], val_labels[j])[1])
    
    
    for j in range(total):
        if(predict_label[j] == 1 and val_labels[j] == 1):
            tp += 1
        if(predict_label[j] == -1 and val_labels[j] == 1):
            fn += 1
        if(predict_label[j] == -1 and val_labels[j] == -1):
            tn += 1
        if(predict_label[j] == 1 and val_labels[j] == -1):
            fp += 1
    
    print("Accuracy: %f" % ((tp+tn)/(tp+fp+tn+fn)))
    recall = tp/(tp+fn)
    print("Recall: %f" % (recall))
    precision = tp/(tp+fp)
    print("Precision: %f" % (precision))
    print("F1: %f" % (2*precision*recall/(precision+recall)))
    '''
    f2 = open("test.csv").readlines()
    testMax = []
    total = 0
    for line in f2:
        total += 1
        line = line.split(',')
        temp = []
        temp.append(1)
        for word in line[:-1]:
            temp.append(float(word))
        testMax.append(temp)

    testMax = np.array(testMax)
    f3=open("15352071_daisiming_PLA.txt", "w")
    for j in range(total):
        predict = 0
        if sum(w * testMax[j]) > 0:
            predict = 1
        elif sum(w * testMax[j]) < 0:
            predict = -1
        f3.write(str(predict)+'\n')
    '''

if __name__ == "__main__":
    import time
    start = time.time()
    iteration = 4000
    Predict(PLA(iteration))
    end = time.time()
    print("%f s" % (end-start))
    



