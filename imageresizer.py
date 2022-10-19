import os
import shutil
import random
import sys
from PIL import Image
import numpy as np
from skimage.io import imread, imsave
from scipy import ndimage
import matplotlib.pyplot as plt




def moveImages(srcDir, destDir):
        """ 
            Code to extract raw data from file system 
        """
        dirList = os.listdir(srcDir)
        imageNames = ("Elyssa-204", "Elyssa-205", "Elyssa-208", "Elyssa-213")
        
        for directory in dirList:
            subdirList = os.listdir(os.path.join(srcDir, directory))
            
            for subdirectory in subdirList:
                artist = subdirectory[4:]
                
                if imageName in imageNames:
                    _moveImageHelper(srcDir, directory, subdirectory, 
                                                       imageName, destDir)

def _moveImageHelper(srcDir, directory, subdirectory, artist, destDir):
        """ 
            Helper function to copy files to new directory 
        """
        files = os.listdir(os.path.join(srcDir, directory, subdirectory))
        for filename in files:
                fullFileName = os.path.join(srcDir, directory, subdirectory, filename)
                if os.path.isfile(fullFileName):
                    shutil.copy(fullFileName, (destDir + '\/' + str(imageName)))

def preprocessImages(srcDir, destDir):
        """ 
            Function to preprocess images into a workable format
            We do NOT want grayscaled
            Reszied
        """
        dirList = os.listdir(srcDir)
        #artists = ("Don_Simone_Camaldolese", "Jacopo_di_Cione", "Lorenzo_Monaco", "Matteo_di_Pacino", "Niccolo_Gerini")
        
        for imageName in dirList:
            
            #os.mkdir(os.path.join(destDir, artist))
            
           #imgList = os.listdir(os.path.join(srcDir, artist))
           #print(len(imgList))
            _helpPreprocessImage(srcDir, imageName, imgList, destDir)
     
def _helpPreprocessImage(srcDir, artist, imgList, destDir):
        """ 
            Helper function to preprocess images into workable format
            DO NOT Grayscale
            Resized
        """
        imgList = os.listdir(os.path.join(srcDir))
        for itrImage in imgList:
                imagePath = os.path.join(srcDir, itrImage)
                openImage = Image.open(imagePath)
                resizedImage = openImage.resize((250,250))
                resizedImage.save(os.path.join(destDir, itrImage))