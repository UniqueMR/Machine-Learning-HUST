'''
数据加载
'''
def loadDataSet(filePath):
    f = open(filePath)
    dataList = []
    labelList = []
    for line in f.readlines():
        currLine = line.strip().split('	')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        dataList.append(lineArr)
        labelList.append(float(currLine[21]))
    return dataList, labelList