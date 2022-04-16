from keras import models,layers

"""构建三层全连接神经网络"""
def modeling(optimizer, loss):
    # 定义Sequential类
    model = models.Sequential()
    # 全连接层，128个节点
    model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
    # 全连接层，64个节点
    model.add(layers.Dense(64, activation='relu'))
    # 全连接层，得到输出
    model.add(layers.Dense(20, activation='softmax'))
    # loss
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    return model