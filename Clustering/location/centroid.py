#随机生成簇中心函数
#k为超参数，代表簇中心的数量
def randCent(dataSet, k):
    n = shape(dataSet)[1] #n=2
    centroids = mat(zeros((k,n)))
    #生成簇中心
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + \
            rangeJ * random.rand(k,1))
    return centroids