import numpy as np
from sklearn.neural_network import MLPClassifier

def dataloader(url):
    '''
    读取文本
    '''
    with open(url, "rb") as fr:
        #以空格为单位对文本进行划分
        data_n = [inst.decode().strip().split(' ') for inst in fr.readlines()]
        data = [[int(element) for element in line] for line in data_n]
    return np.array(data)


def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵
    param:
        sequences: 读取得到的所有文档（词汇编号格式）
        dimension: 词汇表容量，默认值为10000
    """
    #创建矩阵存储每篇文章的词向量
    results = np.zeros((len(sequences), dimension))
    #使用词集模型构建词向量 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 #将出现的词语的索引位置为1
    return results

#载入数据集与测试集
train_data = dataloader("data/train/train_data.txt")
train_labels = dataloader("data/train/train_labels.txt")
test_data = dataloader("data/test/test_data.txt")

#构建词向量
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')

#构建伯努利贝叶斯分类器，进行训练与预测
model = MLPClassifier()
model.fit(x_train, y_train)
y_test = model.predict(x_test)
print('Training Score: %.2f' % model.score(x_train,y_train))

#保存预测结果
result = open('./data/result.txt','w')
for item in y_test:
    result.write(str(int(item))+'\n')
result.close()