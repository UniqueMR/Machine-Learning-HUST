import numpy as np 
from logisticRegression import sigmoid

'''
进行分类
'''
def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
