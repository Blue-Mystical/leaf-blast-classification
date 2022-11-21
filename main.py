from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from imutils import paths

import numpy as np
import cv2
import imutils
import os
import argparse

from utils import LoadDataset
from gridSearch import gridSearch

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

rawImages, features, labels = data.load(pathes, verbose = 50, forceResize = force, genLimit = generatesize)

# TODO: ADD PREPROCESSING METHODS

# TODO: THRESHOLDING WITH OTSU'S ALGORITHM

# train test
# , random_state=4
(trainImg, testImg, trainImgLabel, testImgLabel) = train_test_split(
	rawImages, labels, test_size = train_test_split_size)
(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel) = train_test_split(
	features, labels, test_size = train_test_split_size)

print("[INFO] calculating knn algorithm...")
print("[INFO] train-test split ratio: {:.0f}/{:.0f}".format((1 - train_test_split_size) * 100, train_test_split_size * 100))

print("[INFO] calculating best parameters for the classifier...")
gridSearch(trainImg, trainImgLabel, cv = 25, jobs = jobs)
gridSearch(trainFeature, trainFeatureLabel, cv = 25, jobs = jobs)
# note: using feature histogram is much faster and more accurate than raw image

# # test with the raw images
# print("[INFO] evaluating raw image accuracy...")
# model_raw = KNeighborsClassifier(n_neighbors = k, n_jobs = jobs, weights = 'uniform', metric = 'manhattan')
# cv_scores = cross_val_score(model_raw, trainImg, trainImgLabel, cv = 5)
# print("[INFO] cross validation accuracy across 5 predictions: ")
# print(cv_scores)
# print("[INFO] raw image average accuracy: {}".format(np.mean(cv_scores)))

# print("---------------------------------------")

# # test with color histogram
# print("[INFO] evaluating histogram accuracy...")
# model_feat = KNeighborsClassifier(n_neighbors = k, n_jobs = jobs, weights = 'uniform', metric = 'manhattan')
# cv_scores = cross_val_score(model_feat, trainFeature, trainFeatureLabel, cv = 5)
# print("[INFO] cross validation accuracy across 5 predictions: ")
# print(cv_scores)
# print("[INFO] raw image average accuracy: {}".format(np.mean(cv_scores)))