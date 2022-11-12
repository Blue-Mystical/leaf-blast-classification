from sklearn.neighbors import KNeighborsClassifier
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

# get the list of images
print("[INFO] describing images...")
data = LoadDataset()

label = ['Healthy','LeafBlast']

data.load(pathes, verbose = 5, forceResize = force, genLimit = generatesize)
# rawImages, features, labels = data.load(pathes, verbose = 5, forceResize = force)