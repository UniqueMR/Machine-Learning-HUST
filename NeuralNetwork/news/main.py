import pickle
from vectorizer import vectorizing
from modeling import modeling
from sklearn.model_selection import train_test_split
from evaluation import evaluation
from predRes import predRes

# 读入数据
file_name = './data/train/train_texts.dat'
with open(file_name, 'rb') as f:
    train_texts = pickle.load(f)
file_name = './data/test/test_texts.dat'
with open(file_name, 'rb') as f:
    test_texts = pickle.load(f)

train_labels = []
fl = open('./data/train/train_labels.txt')
for line in fl.readlines():
    train_labels.append(line)

#TF-IDF向量化
train_vector,test_vector,one_hot_train_labels \
    = vectorizing(train_texts,test_texts,train_labels)

#神经网络模型构建
model = modeling(optimizer='rmsprop',loss='categorical_crossentropy')

#拆分测试集与训练集
#由于任务所给测试集中没有标签，因此从原训练集中分割一部分用于测试
X_train, X_test, y_train, y_test = train_test_split\
    (train_vector, one_hot_train_labels, test_size=0.2, random_state=0)
x_test = X_test.toarray()
partial_x_train = X_train.toarray()

#模型训练
history = model.fit(partial_x_train,
                y_train,
                epochs=20,
                batch_size=512,
                validation_data=(x_test, y_test))
#模型测试
evaluation(history)

#使用模型对测试集进行预测
predRes(test_vector,model)

