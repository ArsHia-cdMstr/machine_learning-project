from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
iris = datasets.load_iris()

x = iris.data
y = iris.target


print(x.shape)
print(y.shape)
print(np.unique(y))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)

from sklearn.model_selection import train_test_split

x_train , _ , y_train , y_test = train_test_split(x,y,random_state=42,test_size=0.2)
kmeans.fit(x_train)
x_pred = kmeans.predict(x_train)
print(x_pred.shape)


