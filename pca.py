from sklearn.datasets import fetch_mldata
from sklearn import model_selection, svm, metrics

mnist = fetch_mldata('MNIST original', data_home="./")
mnist_data = mnist.data /255
mnist_label = mnist.target
data_train, data_test, label_train, label_test = model_selection.train_test_split(mnist_data, mnist_label, test_size=100, train_size=1000)
clf = svm.SVC()
clf.fit(data_train, label_train)
pre = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, pre)
print(ac_score)
