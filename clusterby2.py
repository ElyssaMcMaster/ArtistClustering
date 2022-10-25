import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
import resizeimages
style.use("ggplot")

from sklearn.cluster import KMeans

X = resizeimages.imagearray
X= np.array(X).reshape(resizeimages.IMG_SIZE, resizeimages.IMG_SIZE)
print(X)

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)

centroids = kmeans.cluster_centers_ ## the most ideal datapoint based on the datapoints we have; cluster based on equal variance/degrees of variance
labels = kmeans.labels_

print(centroids)
print(labels)
colors = ["g.","r.","c."]


for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize =10)

plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s=150, linewidths = 5, zorder=10)
plt.show()


