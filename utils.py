import sys
import shutil
import os
import cv2
import numpy as np
import imutils

def extract_color_histogram(image, histSize=(8, 8, 8)):
	
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, histSize,
		[0, 180, 0, 256, 0, 256])
	# normalizing the histogram for OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# for OpenCV 3 
	else:
		cv2.normalize(hist, hist)
	# flatten the histogram as the feature vector
	return hist.flatten()

class LoadDataset:
    def __init__(self, width = 64, height = 64):
        self.width = width
        self.height = height

    def load(self, pathes, verbose = -1, forceResize = False, genLimit = -1):

        resizedPath = 'Resized'

        # initialize the raw intensities matrix, the features matrix and labels list
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

            for folder in mainfolder:
                fullmainpath = os.path.join(pathes, folder)
                listfiles = os.listdir(fullmainpath)

                # for display purpose
                genSize = min(len(listfiles), genLimit)

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
        else:
            print("[INFO] Resized folder already found.")

        resizedFolder = os.listdir(resizedPath)

        for folder in resizedFolder:
            fullpath = os.path.join(resizedPath, folder)
            listfiles = os.listdir(fullpath)

            if verbose > 0:
                print('[INFO] loading ', folder, ' ...')
                
            for (i, imagefile) in enumerate(listfiles):

                imagepath = resizedPath + '/' + folder + '/' + imagefile

                image = cv2.imread(imagepath)

                label = folder

                flattenedImage = image.flatten()
                hist = extract_color_histogram(image)

                rawImages.append(flattenedImage)
                features.append(hist)
                labels.append(label)

                # display every n image
                if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                    print("[INFO] loaded {}/{}".format(i, len(listfiles)))

        # calculate size
        rawImages = np.array(rawImages)
        features = np.array(features)
        labels = np.array(labels)
        print("[INFO] raw image list size: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
        print("[INFO] feature list size:   {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))

        return (np.array(rawImages),np.array(features),np.array(labels))