#dataSet为数据集，k为分簇数目，distEclud为距离函数
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    #将所有样本的均值作为簇中心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #创建簇中心列表
    centList =[centroid0] 
    #计算每个样本的误差
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
    #拆分每个簇，并计算拆分后的SSE，选择拆分后SSE最小的簇，保存拆分
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = \
                sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
    #一个簇拆分为二后，其中一个簇新增加簇索引，另一个保存原簇索引号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
    #重置簇中心
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
    #调整样本的簇索引号及误差
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0]\
            ,:]= bestClustAss
    return mat(centList), clusterAssment