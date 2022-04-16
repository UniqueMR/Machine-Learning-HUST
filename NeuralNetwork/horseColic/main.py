from dataloader import loadDataSet
from optimizer1 import stocGradAscent0
from testModel import colicTest 

if __name__ == '__main__':
    numTests = 10
    accSum = 0.0
    trainDataList, trainLabelList \
        = loadDataSet("./data/horseColicTraining.txt")
    testDataList, testLabelList \
        = loadDataSet("./data/horseColicTest.txt")
    for i in range(numTests):
        trainWeights = stocGradAscent0(trainDataList, trainLabelList, 500)
        accSum += colicTest(trainWeights, testDataList, testLabelList)
    print("这%d次的平均精度为%f"%(numTests, accSum/numTests))
