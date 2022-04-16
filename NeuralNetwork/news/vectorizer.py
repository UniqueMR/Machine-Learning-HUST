from sklearn. feature_extraction.text import TfidfVectorizer
from keras.utils.np_utils import to_categorical

def vectorizing(train_texts, test_texts, train_labels):
    """TF-IDF向量化"""
    vectorizer = TfidfVectorizer(max_features=10000)
    train_vector = vectorizer.fit_transform(train_texts)
    test_vector = vectorizer.transform(test_texts)
    one_hot_train_labels = to_categorical(train_labels)
    return train_vector,test_vector,one_hot_train_labels