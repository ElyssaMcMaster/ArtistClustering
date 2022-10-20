import tensorflow as tf 
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)




'''
import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2

DATADIR = "/Users/elyssamcmaster/Desktop/JacopoAndNiccolo"
CATEGORIES = ["Jacopo", "Niccolo"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break
'''