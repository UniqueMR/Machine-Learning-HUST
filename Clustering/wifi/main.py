import numpy as np
import pandas as pd
from processData import process
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

train_data_1 = pd.read_csv('./data/DataSetKMeans1.csv') #原始训练数据
train_data_2 = pd.read_csv('./data/DataSetKMeans2.csv')

BSSID1 = train_data_1['BSSIDLabel'] #获取所有出现的BSSID
bssid1 = set(BSSID1) #统计BSSID的所有可能取值

BSSID2 = train_data_2['BSSIDLabel'] #获取所有出现的BSSID
bssid2 = set(BSSID2) #统计BSSID的所有可能取值

#处理得到新的数据集和测试集
train_dataset_1 = np.array(process(train_data_1, bssid1))
train_dataset_2 = np.array(process(train_data_2, bssid2))

print('DataSetKMeans1:')
for k in range(2,7):
    Kmeans = KMeans(n_clusters=k).fit(train_dataset_1)
    score = davies_bouldin_score(train_dataset_1,Kmeans.labels_)
    print('k='+str(k)+';score='+str(score))

print('DataSetKMeans2:')
for k in range(2,7):
    Kmeans = KMeans(n_clusters=k).fit(train_dataset_2)
    score = davies_bouldin_score(train_dataset_2,Kmeans.labels_)
    print('k='+str(k)+';score='+str(score))