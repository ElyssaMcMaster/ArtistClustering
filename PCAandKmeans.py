import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.image import imread
import random
from tqdm import tqdm


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import plotly as py
import plotly.graph_objs as go
import plotly.express as px

DATADIR= "/Users/elyssamcmaster/Desktop/eyepaintings/"

CATEGORIES = ['JacopoEyes', 'NiccoloEyes']

trainingdata=[]
new_trainingdata = []

IMG_SIZE = 50

def create_training_data():
  for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    class_num= CATEGORIES.index(category)
    for img in tqdm(os.listdir(path)):
        try:
            img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            #print(img_array)
            new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
            #print("the shape is", img_array.shape)
            trainingdata.append([new_array, class_num])
        except Exception as e:
            pass


create_training_data()
#print(len(trainingdata))

random.shuffle(trainingdata)


X = []
y = []

for features,label in trainingdata:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
#print(X[0].shape)
#print(y[0])

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
#print(X.shape)
X_train = X[:80]
y_train = y[:80]
X_test=X[81:]
y_test=y[81:]

#print(X_train.shape)
X = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2])
print(X.shape)

y=np.array(y_train)
print(y.shape)
'''
n=0
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary)
plt.show()
y[n]
'''
Clus_dataSet = StandardScaler().fit_transform(X)
variance = 0.98 #The higher the explained variance the more accurate the model will remain
pca = PCA(variance)

#fit the data according to our PCA instance
pca.fit(Clus_dataSet)

print("Number of components before PCA  = " + str(X.shape[1]))
print("Number of components after PCA 0.98 = " + str(pca.n_components_))


#Transform our data according to our PCA instance
Clus_dataSet = pca.transform(Clus_dataSet)

print("Dimension of our data after PCA  = " + str(Clus_dataSet.shape)) 

#To visualise the data inversed from PCA
approximation = pca.inverse_transform(Clus_dataSet)
print("Dimension of our data after inverse transforming the PCA  = " + str(approximation.shape))

#image reconstruction using the less dimensioned data
plt.figure(figsize=(8,4));

n = 2 #index value, change to view different data

# Original Image
plt.subplot(1, 2, 1);
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(X.shape[1])+' components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 196 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(Clus_dataSet.shape[1]) +' components', fontsize = 14)
plt.title(str(variance * 100) + '% of Variance Retained', fontsize = 20);
plt.show()
print(y_train[n])


k_means = KMeans(init = "k-means++", n_clusters = 2, n_init = 35)
k_means.fit(Clus_dataSet)
k_means_labels = k_means.labels_ #List of labels of each dataset
print("The list of labels of the clusters are " + str(np.unique(k_means_labels)))

G = len(np.unique(k_means_labels)) #Number of labels

#2D matrix  for an array of indexes of the given label
cluster_index= [[] for i in range(G)]
for i, label in enumerate(k_means_labels,0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue     

Y_clust = [[] for i in range(G)]
for n in range(G):
    Y_clust[n] = y[cluster_index[n]] #Y_clust[0] contains array of "correct" category from y_train for the cluster_index[0]
    assert(len(Y_clust[n]) == len(cluster_index[n])) #dimension confirmation

def counter(cluster):
    unique, counts = np.unique(cluster, return_counts=True)
    label_index = dict(zip(unique, counts))
    return label_index

label_count= [[] for i in range(G)]
for n in range(G):
    label_count[n] = counter(Y_clust[n])

label_count[1] #Number of items of a certain category in cluster 1

class_names = {0:'Jacopo', 1:'Niccolo'} #Dictionary of class names

#A function to plot a bar graph for visualising the number of items of certain category in a cluster
def plotter(label_dict):
    plt.bar(range(len(label_dict)), list(label_dict.values()), align='center')
    a = []
    for i in [*label_dict]: a.append(class_names[i])
    plt.xticks(range(len(label_dict)), list(a), rotation=45, rotation_mode='anchor')

k_means_cluster_centers = k_means.cluster_centers_ #numpy array of cluster centers
k_means_cluster_centers.shape #comes from 10 clusters and 420 features

#cluster visualisation
my_members = (k_means_labels == 3) #Enter different Cluster number to view its 3D plot
my_members.shape
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1,projection='3d')
#Clus_dataSet.shape
#Clus_dataSet[my_members,300].shape
ax.plot(Clus_dataSet[my_members, 0], Clus_dataSet[my_members,1],Clus_dataSet[my_members,2], 'w', markerfacecolor="blue", marker='.',markersize=10)


layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)

colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
trace = [ go.Scatter3d() for _ in range(11)]
for i in range(0,10):
    my_members = (k_means_labels == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
            x=Clus_dataSet[my_members, 0],
            y=Clus_dataSet[my_members, 1],
            z=Clus_dataSet[my_members, 2],
            mode='markers',
            marker = dict(size = 2,color = colors[i]),
            hovertext=index,
            name='Cluster'+str(i),
   
            )

fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]], layout=layout)
    
py.offline.iplot(fig)

