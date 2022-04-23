import SVM_Functions as svmF
import numpy as np
from sklearn import svm
from scipy.io import loadmat

x_train, y_train = svmF.loadData('./data/task3_train.mat')
x_test = loadmat('./data/task3_test.mat')['X']
shape = np.shape(x_train)
print('训练集样本数:%d,特征维度:%d' % (shape[0], shape[1]))

'''
output:
训练集样本数:4000,特征维度:1899
'''

c = 1
sigma = 1
# clf = svm.SVC(c, kernel='linear')
clf = svm.SVC(c, kernel='rbf',gamma=sigma)
clf.fit(x_train, y_train)

p = clf.predict(x_train)
r = y_train[:,0]
print('Training Accuracy: {}'.format(np.mean(p == r) * 100))

# pred = clf.predict(x_test)
# res = open('./data/result.txt',mode='a')
# for item in pred:
#     res.write(str(item)+'\n')
# res.close()