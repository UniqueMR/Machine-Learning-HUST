import numpy as np
from logisticRegression import sigmoid

'''
改进的随机梯度下降算法。自定义迭代次数，动态学习率，随机抽样进行训练
'''
def stocGradAscent1(dataList, labelList, numIter=150):
    dataArr = np.array(dataList)
    m,n = np.shape(dataArr)
    weights = np.ones(n) #初始化权重为1
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01 #动态学习率
            #随机抽样
            rand = int(np.random.uniform(0, len(dataIndex)))
            choseIndex = dataIndex[rand]
            #预测值
            h = sigmoid(np.sum(dataArr[choseIndex]*weights))
            #误差
            error = h-labelList[choseIndex]
            #随机梯度下降更新权重
            weights = weights-alpha*error*dataArr[choseIndex]
            #不放回抽样
            del(dataIndex[rand])
    return weights
