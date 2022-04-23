import SVM_Functions as svmF

x,y = svmF.loadData('./data/task1_linear.mat')
svmF.plotData(x, y)

model = svmF.svmTrain_SMO(x, y, C=1, max_iter=20)
svmF.visualizeBoundaryLinear(x, y, model)