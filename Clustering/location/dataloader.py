def loadDataSet(filename):
    datList = []
    #导入数据，获取数据集中的经纬度
    for line in open(filename).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    return datMat