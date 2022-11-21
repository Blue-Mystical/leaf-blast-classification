from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from imutils import paths

import numpy as np
import cv2
import imutils
import os
import argparse

from utils import LoadDataset

# parse the argument
#--dataset --k --threadcount
#--dataset is required
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="number of nearest neighbors (default is 1)")
ap.add_argument("-t", "--threads", type=int, default=-1,
	help="number of threads for k-NN distance (default is -1 and uses maximum threads)")
ap.add_argument("-f", "--forceresize", type=bool, default=False,
	help="whether to force generate a resized dataset folder even if there's one already")
ap.add_argument("-g", "--generatesize", type=int, default=-1,
	help="how many images per folder to to be resized when it should generate a resized dataset (default to all)")
args = vars(ap.parse_args())

pathes = args["dataset"]
k = args["neighbors"]
force = args["forceresize"]
generatesize = args["generatesize"]
jobs = args["threads"]

if generatesize >= 0 and force == False:
	print("[WARN] the 'generatesize' argument requires 'forceresize' argmument to be True")

# get the list of images
print("[INFO] describing images...")
data = LoadDataset()

train_test_split_size = 0.3

label = ['Healthy','LeafBlast']

rawImages, features, labels = data.load(pathes, verbose = 5, forceResize = force, genLimit = generatesize)

# TODO: ADD PREPROCESSING METHODS

# TODO: THRESHOLDING WITH OTSU'S ALGORITHM


# train test
# , random_state=4
(trainImg, testImg, trainImgLabel, testImgLabel) = train_test_split(
	rawImages, labels, test_size = train_test_split_size)
(trainFeature, testFeature, trainFeatureLabels, testFeatureLabels) = train_test_split(
	features, labels, test_size = train_test_split_size)

# TODO: 

print("[INFO] calculating knn algorithm...")
print("[INFO] train-test split ratio: {:.0f}/{:.0f}".format((1 - train_test_split_size) * 100, train_test_split_size * 100))
model = KNeighborsClassifier(n_neighbors = k, n_jobs=jobs)
# model.fit(trainImg, trainImgLabel)

# accuracy = model.score(testImg, testImgLabel)
# print("[INFO] raw accuracy: {:.2f}%".format(accuracy * 100))

cv_scores = cross_val_score(model, trainImg, trainImgLabel, cv = 5)
print("[INFO] cross validation accuracy across 5 predictions: ")
print(cv_scores)
print("[INFO] average accuracy: {}".format(np.mean(cv_scores)))