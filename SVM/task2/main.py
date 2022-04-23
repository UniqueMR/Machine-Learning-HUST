import SVM_Functions as svmF
import numpy as np
import pandas as pd

x,y = svmF.loadData('./data/task2.mat')
svmF.plotData(x, y)


c = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

acc = np.zeros((len(c),len(sigma)))
cnt = 0

for i,c_item in enumerate(c):
    for j,sigma_item in enumerate(sigma):
        model = svmF.svmTrain_SMO(x, y, C=c_item, kernelFunction\
            ='gaussian',K_matrix=svmF.gaussianKernel(x, sigma=sigma_item))
        p = svmF.svmPredict(X=np.mat(x),model=model,sigma=sigma_item)
        cnt = 0
        for k in range(len(p)):
            if int(p[k]) == y[k]:
                cnt += 1
        accuracy = cnt/len(p)
        acc[i][j] = accuracy

print(acc)
pd.DataFrame(acc).to_csv('./data/result.csv')