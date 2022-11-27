from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import sys

from sklearn.metrics import confusion_matrix, classification_report

class ClassificationModel:

    def __init__(self, modelType = "knn", knn_k = 5, n_jobs = -1):
        if modelType == "knn":
            self.model = KNeighborsClassifier(n_neighbors = knn_k, n_jobs = n_jobs, weights = 'uniform', metric = 'manhattan')
            self.initialized = True
        else:
            print("[ERROR] Invalid model name identifier detected.")

    def train(self, X, y, cv):

        if self.initialized == True:
            # train with the raw images
            print("[INFO] evaluating raw image accuracy...")

            cv_score_accuracy = cross_val_score(self.model, X, y , cv = cv)
            print("[INFO] cross validation accuracy: ")
            print(['%.4f' % elem for elem in cv_score_accuracy])
            print("[INFO] raw image average accuracy: {0:.4}".format(np.mean(cv_score_accuracy)))

            cv_score_recall = cross_val_score(self.model, X, y, cv = cv, scoring='recall')
            print("[INFO] cross validation recall: ")
            print(['%.4f' % elem for elem in cv_score_recall])
            print("[INFO] raw image average recall: {0:.4}".format(np.mean(cv_score_recall)))

            cv_score_precision = cross_val_score(self.model, X, y, cv = cv, scoring='precision')
            print("[INFO] cross validation precision: ")
            print(['%.4f' % elem for elem in cv_score_precision])
            print("[INFO] raw image average precision: {0:.4}".format(np.mean(cv_score_precision)))
        else:
            print("[ERROR] Cannot run improperly-initialized model.")

    def test(self, X_train, X_test, y_train, y_test, labelList):
        if self.initialized == True:
            self.model.fit(X_train, np.ravel(y_train))
            test_accuracy = self.model.predict(X_test)

            print(classification_report(y_test, test_accuracy, target_names=labelList, digits=4))
            print(confusion_matrix(y_test, test_accuracy))
        else:
            print("[ERROR] Cannot run improperly-initialized model.")

