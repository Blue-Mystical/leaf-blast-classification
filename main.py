from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
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

# train test
# , random_state=4
(trainImg, testImg, trainImgLabel, testImgLabel) = train_test_split(
	rawImages, labels, test_size = train_test_split_size)
(trainFeature, testFeature, trainFeatureLabel, testFeatureLabel) = train_test_split(
	features, labels, test_size = train_test_split_size)

print("[INFO] calculating knn algorithm...")
print("[INFO] train-test split ratio: {:.0f}/{:.0f}".format((1 - train_test_split_size) * 100, train_test_split_size * 100))

# print("[INFO] calculating best parameters for the classifier...")
# gridSearch(trainImg, trainImgLabel, cv = 5, jobs = jobs)
# gridSearch(trainFeature, trainFeatureLabel, cv = 5, jobs = jobs)

# note: using feature histogram is much faster and more accurate than raw image

cv = 5

print("----------------[TRAIN RAW IMAGE]------------------")

# test with the raw images
print("[INFO] evaluating raw image accuracy...")
model_raw = KNeighborsClassifier(n_neighbors = k, n_jobs = jobs, weights = 'uniform', metric = 'manhattan')

cv_score_accuracy = cross_val_score(model_raw, trainImg, np.ravel(trainImgLabel) , cv = cv)
print("[INFO] cross validation accuracy: ")
print(cv_score_accuracy)
print("[INFO] raw image average accuracy: {}".format(np.mean(cv_score_accuracy)))

cv_score_recall = cross_val_score(model_raw, trainFeature, np.ravel(trainFeatureLabel), cv = cv, scoring='recall')
print("[INFO] cross validation recall: ")
print(cv_score_recall)
print("[INFO] raw image average recall: {}".format(np.mean(cv_score_recall)))

cv_score_precision = cross_val_score(model_raw, trainFeature, np.ravel(trainFeatureLabel), cv = cv, scoring='precision')
print("[INFO] cross validation precision: ")
print(cv_score_precision)
print("[INFO] raw image average precision: {}".format(np.mean(cv_score_precision)))

print("----------------[TRAIN HISTOGRAM]------------------")

# test with color histogram
print("[INFO] evaluating histogram accuracy...")
model_feat = KNeighborsClassifier(n_neighbors = k, n_jobs = jobs, weights = 'uniform', metric = 'manhattan')

cv_score_accuracy = cross_val_score(model_feat, trainFeature, np.ravel(trainFeatureLabel), cv = cv)
print("[INFO] cross validation accuracy: ")
print(cv_score_accuracy)
print("[INFO] histogram average accuracy: {}".format(np.mean(cv_score_accuracy)))

cv_score_recall = cross_val_score(model_feat, trainFeature, np.ravel(trainFeatureLabel), cv = cv, scoring='recall')
print("[INFO] cross validation recall: ")
print(cv_score_recall)
print("[INFO] histogram average recall: {}".format(np.mean(cv_score_recall)))

cv_score_precision = cross_val_score(model_feat, trainFeature, np.ravel(trainFeatureLabel), cv = cv, scoring='precision')
print("[INFO] cross validation precision: ")
print(cv_score_precision)
print("[INFO] histogram average precision: {}".format(np.mean(cv_score_precision)))

print("----------------[TEST RAW IMAGE]------------------")

model_raw_fit = model_raw.fit(trainImg, np.ravel(trainImgLabel))
test_accuracy = model_raw.predict(testImg)

print(classification_report(testImgLabel, test_accuracy, target_names=data.labelClasses))
print(confusion_matrix(testImgLabel, test_accuracy))

print("----------------[TEST HISTOGRAM]------------------")

model_hist_fit = model_feat.fit(trainFeature, np.ravel(trainFeatureLabel))
test_accuracy = model_feat.predict(testFeature)

print(classification_report(testFeatureLabel, test_accuracy, target_names=data.labelClasses))
print(confusion_matrix(testFeatureLabel, test_accuracy))
