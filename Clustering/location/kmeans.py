#dataSet为数据集，k为分簇数目，
#distEclud为距离函数，randCent为随机选择簇中心方法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    #初始化矩阵clusterAssment，第1列记录簇索引值，第2列存储误差
    clusterAssment = mat(zeros((m,2)))  
    #初始化簇中心                            
    centroids = createCent(dataSet, k)
    #标志变量，用于判断是否继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
    #将每个样本点分配到与其最近的簇中心所在的簇
        for i in range(m): 
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
    #如果样本被划分到不同的簇，则改变标志变量，表示需要继续迭代
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
    # #打印簇中心
    #     print (centroids)
    #由于样本划分发生变化，因此需要重新计算簇中心
        for cent in range(k):
    #提取处属于同一簇的所有样本
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
    #计算不同簇所有样本的平均值作为簇中心
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    #返回最终的聚类中心和聚类结果
    return centroids, clusterAssment