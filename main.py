from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import argparse

from utils import LoadDataset
from gridSearch import gridSearch
from models import ClassificationModel

# parse the argument
#--dataset --k --threadcount
#--dataset is required
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="number of nearest neighbors only for KNN (default is 1)")
ap.add_argument("-t", "--threads", type=int, default=-1,
	help="number of threads (default is -1 and uses maximum threads)")
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

# warns you if you use generatesize when you don't the forceresize option to true
if generatesize >= 0 and force == False:
	print("[WARN] the 'generatesize' argument requires 'forceresize' argmument to be True")

# get the list of images
print("[INFO] describing images...")
data = LoadDataset()

train_test_split_size = 0.3

rawImages, features, labels = data.load(pathes, verbose = 50, forceResize = force, genLimit = generatesize)
labelList = data.labelClasses

# train test
# , random_state=4
(trainImg, testImg, trainImgLabel, testImgLabel) = train_test_split(
	rawImages, labels, test_size = train_test_split_size)
(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel) = train_test_split(
	features, labels, test_size = train_test_split_size)

print("[INFO] train-test split ratio: {:.0f}/{:.0f}".format((1 - train_test_split_size) * 100, train_test_split_size * 100))

# print("[INFO] calculating best parameters for the classifier...")
# gridSearch(trainImg, np.ravel(trainImgLabel), cv = 5, jobs = jobs)
# gridSearch(trainFeature, np.ravel(trainFeatureLabel), cv = 5, jobs = jobs)

# note: using feature histogram is much faster and more accurate than raw image

cv = 5

mode = "knn"

if mode == "knn" or mode == "any":
	knn_raw = ClassificationModel(modelType = "knn", knn_k = k, n_jobs = jobs)
	knn_raw.train(trainImg, np.ravel(trainImgLabel), cv = 5)
	knn_raw.test(trainImg, testImg, trainImgLabel, testImgLabel, labelList)

	knn_hist = ClassificationModel(modelType = "knn", knn_k = k, n_jobs = jobs)
	knn_hist.train(trainFeature, np.ravel(trainFeatureLabel), cv = 5)
	knn_hist.test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)