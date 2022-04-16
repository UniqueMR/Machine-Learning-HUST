import numpy as np
from sklearn.linear_model import LogisticRegression

def colicSklearn():
    #使用np读取数据
    trainFiled=np.loadtxt('./data/horseColicTraining.txt',delimiter="\t")
    trainSet = trainFiled[:,:-1]
    trainLables=trainFiled[:,-1:]

    testFiled = np.loadtxt('./data/horseColicTest.txt', delimiter="\t")
    testSet = testFiled[:, :-1]
    testLables = testFiled[:, -1:]
    accSum = 0.0
    for i in range(10):
        classifier=LogisticRegression(class_weight='balanced',\
            solver='liblinear',max_iter=10).fit(trainSet,trainLables)
        test_accurcy=classifier.score(testSet,testLables) *100
        accSum += test_accurcy
        print('正确率：%f%%'%test_accurcy)
    print('平均正确率：%f%%'% (accSum/10))
if __name__ == '__main__':
    colicSklearn()