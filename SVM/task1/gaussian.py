import SVM_Functions as svmF

x, y = svmF.loadData('./data/task1_gaussian.mat')
svmF.plotData(x, y)

model = svmF.svmTrain_SMO(x, y, C=1, \
    kernelFunction='gaussian', K_matrix=svmF.gaussianKernel(x, sigma=0.1))
svmF.visualizeBoundaryGaussian(x, y, model,sigma=0.1)