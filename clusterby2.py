import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
import resizeimages
from sklearn.decomposition import PCA
#from scipy.misc import imread
#from matplotlib.pyplot import imread
#style.use("ggplot")
from glob import glob
import pandas as pd
import cv2 as cv
import os

#file_names=resizeimages.imagearray
#print(file_names)

'''
eyes=pd.DataFrame([])
for path in file_names:
    print(path)
    #img=cv.imread(path)
    #print(img)
    #eye=pd.Series(img.flatten(),name=path)
    #eyes=eyes.append(eye)


'''


''''
from sklearn.cluster import KMeans

X = resizeimages.imagearray
for item in X:
    np.array(item).reshape(resizeimages.IMG_SIZE, resizeimages.IMG_SIZE)
print(X)

def transform(percentage):
    for item in X:
        percentage = percentage/100
        eye_pca = PCA(n_components=percentage).fit(item)
        transformed = eye_pca.transform(item)
        projection = eye_pca.inverse_transform(transformed)



kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

centroids = kmeans.cluster_centers_ ## the most ideal datapoint based on the datapoints we have; cluster based on equal variance/degrees of variance
labels = kmeans.labels_

print(centroids)
print(labels)
colors = ["g.","r.", "c."]


for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize =10)

plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s=150, linewidths = 5, zorder=10)
plt.show()

'''
