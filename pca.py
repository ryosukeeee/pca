from sklearn.datasets import fetch_mldata
from sklearn import model_selection, svm, metrics
from sklearn.decomposition import PCA
import time
mnist = fetch_mldata('MNIST original', data_home="./")
#主成分分析
pca = PCA(n_components=20)
pca_mnist_data = pca.fit_transform(mnist.data /255)

mnist_data = mnist.data /255
mnist_label = mnist.target
data_train, data_test, label_train, label_test = model_selection.train_test_split(mnist_data, mnist_label, test_size=100, train_size=1000)

start = time.time()
clf = svm.SVC()
clf.fit(data_train, label_train)
pre = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, pre)
print(ac_score)
end = time.time()
print(end-start)

data_train, data_test, label_train, label_test = model_selection.train_test_split(pca_mnist_data, mnist_label, test_size=100, train_size=1000)

start = time.time()
pca_clf = svm.SVC()
pca_clf.fit(data_train, label_train)
pre = pca_clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, pre)
print(ac_score)
end = time.time()
print(end-start)
