import numpy as np
from logisticRegression import sigmoid

'''
随机梯度下降算法。自定义迭代次数，静态学习率，遍历所有数据进行训练
'''
def stocGradAscent0(dataList, labelList, iter=150):
    dataArr = np.array(dataList)  # 数据转化为矩阵
    m, n = np.shape(dataArr)
    alpha = 0.01 #固定学习率
    weights = np.ones(n) #初始化权重为1
    for j in range(iter):
        for i in range(m):
            h = sigmoid(np.sum(dataArr[i]*weights))
            error = (h-labelList[i])
            weights = weights-alpha*error*dataArr[i]
    return weights
