import numpy as np

'''
sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
 