
'''
from PIL import Image
import glob
from numpy import asarray

image_list = []
resized_images = []
numpydata= []

for filename in glob.glob("/Users/elyssamcmaster/Desktop/JacopoAndNiccolo/Images/*.jpg"):
    print(filename)
    img = Image.open(filename)
    image_list.append(img)

for filename in glob.glob("/Users/elyssamcmaster/Desktop/JacopoAndNiccolo/Images/*.png"):
    print(filename)
    img = Image.open(filename)
    
    image_list.append(img)

#for (i, new) in enumerate(resized_images):
 #   new.save('{}{}{}'.format("/Users/elyssamcmaster/Desktop/JacopoAndNiccolo/ResizedImages", i+1, '.png'))


for image in image_list:
    #image.show()
    image = image.resize((400,400))
    resized_images.append(image)


for image in resized_images:
    data=asarray(image)
    numpydata.append(data)
#numpydata = asarray(resized_images)
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


DATADIR= "/Users/elyssamcmaster/Desktop/JacopoAndNiccolo/images"
#CATEGORIES = ["Elyssa-204","Elyssa-205","Elyssa-208","Elyssa-213"]

imagearray=[]

IMG_SIZE = 400
    #path = os.path.join(DATADIR, category) #path to every individual painting
for img in os.listdir(DATADIR):
    img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE) #grayscale to account for constraints of being on a laptop
    #print(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(new_array, cmap='gray')
    #plt.show()
    imagearray.append(new_array)
    break




