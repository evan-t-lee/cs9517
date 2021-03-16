import tensorflow as tf
import numpy as np

from sklearn import neighbors, linear_model, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

models = {
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=3),
    'SGD': linear_model.SGDClassifier(max_iter=250),
    'DT': tree.DecisionTreeClassifier()
}

train_size = 2000
test_size = 500

def train_and_test(classifier):
    model = models[classifier].fit(train_data, train_labels)
    results = model.predict(test_data)

    accuracy = accuracy_score(test_labels, results)
    print(f'Accuracy score: {accuracy}')

    recall = recall_score(test_labels, results, average='macro')
    print(f'Recall score: {recall}')

    confusion = confusion_matrix(test_labels, results)
    print(f'Confusion Matrix:\n{confusion}')

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
data = np.concatenate((train_data, test_data), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_size, test_size=test_size)
print(train_data.shape, test_data.shape)
train_data = np.reshape(train_data, (train_size, -1))
test_data = np.reshape(test_data, (test_size, -1))
print(train_data.shape, test_data.shape)