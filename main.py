import preprocessing
import twoartistcnn
import os

def main():
    splits = 10

        
    srcDir = (r'/Users/elyssamcmaster/Desktop/eyes/alleyes')


    destDir = (r'/Users/elyssamcmaster/Desktop/eyes/suppdata')

    preprocessing.generateSyntheticData(srcDir, destDir)  

if __name__== "__main__":
    main()