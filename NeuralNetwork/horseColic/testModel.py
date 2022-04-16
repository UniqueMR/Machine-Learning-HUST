import numpy as np
from classifier import classifyVector

'''
在测试集上计算分类精度
'''
def colicTest(trainWeights, testDataList, testLabelList):
    rightCount = 0  # 判断错误的数量
    testCount = len(testDataList)
    for i in range(testCount):
        if int(classifyVector(np.array(testDataList[i]),\
             trainWeights))==int(testLabelList[i]):
            rightCount += 1
    acc = float(rightCount)/testCount
    print("本次的精度为%f" % acc)
    return acc