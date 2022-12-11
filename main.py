from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import collections

import numpy as np
import argparse

from utils import LoadDataset
from gridSearch import gridSearch
from models import ClassificationModel

# parse the argument
#--dataset is required
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="all",
	help="available models: all, knn, rdf, dt, svm, nb (default to all)")
ap.add_argument("-t", "--threads", type=int, default=-1,
	help="number of threads for knn, rdf and svm (default is -1 and uses maximum threads)")
ap.add_argument("-f", "--forceresize", type=bool, default=False, action=argparse.BooleanOptionalAction,
	help="whether to force generate a resized dataset folder even if there's one already [Must use --dataset or -d followed by a folder's name afterward]")
ap.add_argument("-d", "--dataset",
	help="path to input dataset for regenerating resized folder")
ap.add_argument("-g", "--generatesize", type=int, default=-1,
	help="how many images per subfolder to to be resized when it should generate a resized dataset (default to all)")
ap.add_argument("-s", "--gridsearch", type=bool, default=False, action=argparse.BooleanOptionalAction,
	help="use grid search to fine tune parameters")
ap.add_argument("-i", "--displayimage", type=bool, default=False, action=argparse.BooleanOptionalAction,
	help="sample an image of both raw and preprocessed")
args = vars(ap.parse_args())

mode = args["model"].lower()
force = args["forceresize"]
pathes = args["dataset"]
generatesize = args["generatesize"]
jobs = args["threads"]
displayImage = args["displayimage"]
GridSearchMode = args["gridsearch"]

# warns you if you use generatesize when you don't the forceresize option to true
if generatesize >= 0 and force == False:
	print("[WARN] The 'generatesize' argument requires 'forceresize' argmument to be True")

# get the list of images
print("[INFO] describing images...")
data = LoadDataset(width = 256, height = 256, displayImage = displayImage)

rawImages, features, labels = data.load(pathes, verbose = 50, forceResize = force, genLimit = generatesize)
labelList = data.labelClasses

# train test split
train_test_split_size = 0.2

(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel) = train_test_split(
	features, labels, test_size = train_test_split_size)

print("[INFO] train-test split ratio: {:.0f}/{:.0f}".format((1 - train_test_split_size) * 100, train_test_split_size * 100))


# SMOTE oversampling
oversample = False

if oversample == True:
	oversampling_size = 1.0

	print("[INFO] Oversampling data...")
	print("[INFO] Normal train data count: ", collections.Counter(np.ravel(trainFeatureLabel)))

	sm = SMOTE(random_state = 5, sampling_strategy=oversampling_size)
	trainFeature, trainFeatureLabel = sm.fit_resample(trainFeature, trainFeatureLabel)

	print("[INFO] Oversampled train data count: ", collections.Counter(np.ravel(trainFeatureLabel)))

# Main Program
cv = 5

if GridSearchMode == False:
	if mode == "knn" or mode == "all":
		# warning: using raw image dataset will take a massive amount of time especially in RF
		# knn_raw = ClassificationModel(modelType = "knn", knn_k = k, n_jobs = jobs)
		# knn_raw.train(trainImg, np.ravel(trainImgLabel), cv = 5)
		# knn_raw.test(trainImg, testImg, trainImgLabel, testImgLabel, labelList)

		# NOTE: cross_validate is not used to fit the model. train_test is used to fit to test the result.

		print("[INFO] -----------------KNN CLASSIFIER-----------------")
		knn_hist = ClassificationModel(modelType = "knn", n_jobs = jobs)
		knn_hist.cross_validate(trainFeature, np.ravel(trainFeatureLabel), cv = cv)
		knn_hist.train_test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)

	if mode == "rf" or mode == "randomforest" or mode == "all":
		print("[INFO] -----------------RANDOM FOREST CLASSIFIER-----------------")
		knn_hist = ClassificationModel(modelType = "rf", n_jobs = jobs)
		knn_hist.cross_validate(trainFeature, np.ravel(trainFeatureLabel), cv = cv)
		knn_hist.train_test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)

	if mode == "dt" or mode == "decisiontree" or mode == "all":
		print("[INFO] -----------------DECISION TREE CLASSIFIER-----------------")
		knn_hist = ClassificationModel(modelType = "dt", n_jobs = jobs)
		knn_hist.cross_validate(trainFeature, np.ravel(trainFeatureLabel), cv = cv)
		knn_hist.train_test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)

	if mode == "nb" or mode == "naivebayes" or mode == "all":
		print("[INFO] -----------------NAIVE BAYES CLASSIFIER-----------------")
		knn_hist = ClassificationModel(modelType = "nb", n_jobs = jobs)
		knn_hist.cross_validate(trainFeature, np.ravel(trainFeatureLabel), cv = cv)
		knn_hist.train_test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)

	if mode == "svc" or mode == "svm" or mode == "all":
		print("[INFO] -----------------SVM CLASSIFIER-----------------")
		knn_hist = ClassificationModel(modelType = "svm", n_jobs = jobs)
		knn_hist.cross_validate(trainFeature, np.ravel(trainFeatureLabel), cv = cv)
		knn_hist.train_test(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel, labelList)
else:
	print("[INFO] calculating best parameters for the model(s)...")

	if mode == "knn" or mode == "all":

		print("[INFO] -----------------KNN CLASSIFIER-----------------")
		# gridSearch(rawImages, np.ravel(labels), cv = cv, jobs = jobs)
		gridSearch(trainFeature, np.ravel(trainFeatureLabel), modelType = "knn", cv = cv, jobs = jobs)


	if mode == "rf" or mode == "randomforest" or mode == "all":
		print("[INFO] -----------------RANDOM FOREST CLASSIFIER-----------------")
		gridSearch(trainFeature, np.ravel(trainFeatureLabel), modelType = "rf", cv = cv, jobs = jobs)

	if mode == "dt" or mode == "decisiontree" or mode == "all":
		print("[INFO] -----------------DECISION TREE CLASSIFIER-----------------")
		gridSearch(trainFeature, np.ravel(trainFeatureLabel), modelType = "dt", cv = cv)

	if mode == "nb" or mode == "naivebayes" or mode == "all":
		print("[INFO] -----------------NAIVE BAYES CLASSIFIER-----------------")
		gridSearch(trainFeature, np.ravel(trainFeatureLabel), modelType = "nb", cv = cv)

	if mode == "svc" or mode == "svm" or mode == "all":
		print("[INFO] -----------------SVM CLASSIFIER-----------------")
		gridSearch(trainFeature, np.ravel(trainFeatureLabel), modelType = "svm", cv = cv, jobs = jobs)
