import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


DATADIR= "/Users/elyssamcmaster/Desktop/JacopoAndNiccolo/allimages"


imagearray=[]

IMG_SIZE = 300
for img in os.listdir(DATADIR):
    img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE) #grayscale to account for constraints of being on a laptop
    #print(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(new_array, cmap='gray')
    #plt.show()
    imagearray.append(new_array)
    print(imagearray)
    break




