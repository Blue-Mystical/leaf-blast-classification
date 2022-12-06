import shutil
import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from sklearn import preprocessing

from time import time

def extract_color_histogram(image, histSize=(8, 8, 8)):

    # contrasting
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))

    contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, histSize,
        [0, 180, 0, 256, 0, 256])

    # normalizing the histogram for OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    # for OpenCV 3 
    else:
        cv2.normalize(hist, hist)

    # masking (not used atm)
    lower_mask = np.array([0, 60, 180])
    upper_mask = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_mask, upper_mask)

    # Change masked spot to light yellow
    result = image.copy()
    result[mask > 0] = (113, 200, 200)

    hsv_mask = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    hist_mask = cv2.calcHist([hsv_mask], [0, 1, 2], None, histSize,
        [0, 180, 0, 256, 0, 256])

    # returns the histogram which will flatten as the feature vector later
    return hsv, hist, mask, result, hsv_mask, hist_mask

class LoadDataset:

    labelClasses = []

    # init values
    def __init__(self, width = 256, height = 256, displayImage = False):
        self.width = width
        self.height = height
        self.displayImage = displayImage
        self.labelClasses = []

    def load(self, pathes, verbose = -1, forceResize = False, genLimit = -1):

        resizedPath = 'Resized'

        # initialize the raw image matrix, the features matrix and labels list
        rawImages = []
        features = []
        labels = []

        mainfolder = os.listdir(pathes)

        # initialize by generating a new folder of resized image if there's none or forced to
        isExist = os.path.exists(resizedPath)

        if isExist == False or forceResize == True:
            if isExist and os.path.isdir(resizedPath):
                print("[INFO] Removing existing folder...")
                shutil.rmtree(resizedPath)
            
            print("[INFO] Regenerating resized dataset...")
            os.makedirs(resizedPath)

            startTime = time()

            for folder in mainfolder:
                fullmainpath = os.path.join(pathes, folder)
                listfiles = os.listdir(fullmainpath)

                # for displaying file count
                if genLimit > -1:
                     genSize = min(len(listfiles), genLimit)
                else:
                     genSize = len(listfiles)

                fullresizedfolder = os.path.join(resizedPath, folder)
                os.makedirs(fullresizedfolder)

                if verbose > 0:
                    print('[INFO] resizing ', folder, ' ...')
                    
                for (i, imagefile) in enumerate(listfiles):

                    imagepath = pathes + '/' + folder + '/' + imagefile

                    # read dataset image then write a new resized image on the other folder
                    image = cv2.imread(imagepath)

                    resizedImage = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(fullresizedfolder, imagefile), resizedImage)

                    # display every n image
                    if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                        print("[INFO] resized {}/{}".format(i, genSize))
                        
                    if genLimit >= 0:
                        if i >= genLimit - 1:
                            break
            print("[INFO] Resize Completed.")
            
            endTime = time()
            elapsed = endTime - startTime
            print('[INFO] Time taken to resize images: %f seconds.' % elapsed)
        else:
            print("[INFO] Resized folder already found.")

        # loads resized folder
        startTime = time()

        resizedFolder = os.listdir(resizedPath)

        for folder in resizedFolder:
            fullpath = os.path.join(resizedPath, folder)
            listfiles = os.listdir(fullpath)

            if verbose > 0:
                print('[INFO] loading ', folder, ' ...')
                
            for (i, imagefile) in enumerate(listfiles):

                imagepath = resizedPath + '/' + folder + '/' + imagefile

                # load image and generates a color histogram from that image
                image = cv2.imread(imagepath)
                hsv, hist, mask, result, hsv_mask, hist_mask = extract_color_histogram(image)

                # flatten images 
                flattenedImage = image.flatten()
                flattenedHist = hist.flatten()
                flattenedHistMask = hist_mask.flatten()
                label = folder

                # display images
                if i == 1 and self.displayImage == True:

                    cv2.namedWindow(folder + ' raw image', cv2.WINDOW_NORMAL)
                    cv2.imshow(folder + " raw image", image)
                    cv2.namedWindow(folder + ' hsv image', cv2.WINDOW_NORMAL)
                    cv2.imshow(folder + " hsv image", hsv)
                    # cv2.namedWindow(folder + ' mask', cv2.WINDOW_NORMAL)
                    # cv2.imshow(folder + " mask", mask)
                    # cv2.namedWindow(folder + ' masked raw', cv2.WINDOW_NORMAL)
                    # cv2.imshow(folder + " masked raw", result)
                    # cv2.namedWindow(folder + ' hsv masked', cv2.WINDOW_NORMAL)
                    # cv2.imshow(folder + " hsv masked", hsv_mask)
                    print("Press any key while focusing on the image window to continue.")
                    cv2.waitKey() 

                    # print(flattenedHist)
                    # plt.plot(flattenedHist)
                    # plt.show()

                # append the lists
                rawImages.append(flattenedImage)
                features.append(flattenedHist)
                labels.append(label)

                # display every n image
                if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                    print("[INFO] loaded {}/{}".format(i, len(listfiles)))

        # binarize the label list
        lb = preprocessing.LabelBinarizer()
        binaryLabels = lb.fit_transform(labels)
        self.labelClasses = lb.classes_
        print("[INFO] Available classes: ", lb.classes_)

        # calculate array size in MB
        rawImages = np.array(rawImages)
        features = np.array(features)
        labels = np.array(binaryLabels)
        print("[INFO] raw image list size: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
        print("[INFO] feature list size:   {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))
        print("[INFO] labels list size:   {:.2f}MB".format(labels.nbytes / (1024 * 1000.0)))

        endTime = time()
        elapsed = endTime - startTime
        print('[INFO] Time taken to load and preprocess images: %f seconds.' % elapsed)

        return (np.array(rawImages),np.array(features),np.array(labels))