from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
from time import time

from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

class ClassificationModel:

    def __init__(self, modelType = "knn", n_jobs = -1):
        if modelType == "knn":
            self.model = KNeighborsClassifier(n_neighbors = 5, n_jobs = n_jobs, weights = 'uniform', metric = 'manhattan')
            self.initialized = True
        elif modelType == "rf"  or modelType == "randomforest":
            self.model = RandomForestClassifier(n_jobs= n_jobs, n_estimators=100)
            self.initialized = True
        elif modelType == "dt"  or modelType == "decisiontree":
            self.model = DecisionTreeClassifier()
            self.initialized = True
        elif modelType == "nb" or modelType == "naivebayes":
            self.model = GaussianNB()
            self.initialized = True
        elif modelType == "svm" or modelType == "svc":
            self.model = svm.SVC()
            self.initialized = True
        else:
            print("[ERROR] Invalid model name identifier detected.")

    def cross_validate(self, X, y, cv):

        if self.initialized == True:
            print("[INFO] evaluating raw image accuracy...")
            startTime = time()

            cv_results = cross_validate(self.model, X, y , cv = cv, scoring=scoring)

            print("[INFO] VAL_ACCURACY\t:", ['%.4f' % elem for elem in cv_results["test_accuracy"]])
            print("[INFO] VAL_PRECISION\t:", ['%.4f' % elem for elem in cv_results["test_precision"]])
            print("[INFO] VAL_RECALL\t:", ['%.4f' % elem for elem in cv_results["test_recall"]])
            print("[INFO] VAL_F1SCORE\t:", ['%.4f' % elem for elem in cv_results["test_f1_score"]])

            print("[INFO] --------------------------------------")

            print("[INFO] AVG_ACCURACY\t: {0:.4}".format(np.mean(cv_results["test_accuracy"])))
            print("[INFO] AVG_PRECISION\t: {0:.4}".format(np.mean(cv_results["test_precision"])))
            print("[INFO] AVG_RECALL\t: {0:.4}".format(np.mean(cv_results["test_recall"])))
            print("[INFO] AVG_F1SCORE\t: {0:.4}".format(np.mean(cv_results["test_f1_score"])))
                
            # print("[INFO] cross validation accuracy: ")
            # print(['%.4f' % elem for elem in cv_score_accuracy])
            # print("[INFO] raw image average accuracy: {0:.4}".format(np.mean(cv_score_accuracy)))

            # cv_score_accuracy = cross_val_score(self.model, X, y , cv = cv)
            # print("[INFO] cross validation accuracy: ")
            # print(['%.4f' % elem for elem in cv_score_accuracy])
            # print("[INFO] raw image average accuracy: {0:.4}".format(np.mean(cv_score_accuracy)))

            # cv_score_recall = cross_val_score(self.model, X, y, cv = cv, scoring='recall')
            # print("[INFO] cross validation recall: ")
            # print(['%.4f' % elem for elem in cv_score_recall])
            # print("[INFO] raw image average recall: {0:.4}".format(np.mean(cv_score_recall)))

            # cv_score_precision = cross_val_score(self.model, X, y, cv = cv, scoring='precision')
            # print("[INFO] cross validation precision: ")
            # print(['%.4f' % elem for elem in cv_score_precision])
            # print("[INFO] raw image average precision: {0:.4}".format(np.mean(cv_score_precision)))

            endTime = time()
            elapsed = endTime - startTime
            print('[INFO] Time taken: %f seconds.' % elapsed)
        else:
            print("[ERROR] Cannot run improperly-initialized model.")

    def train_test(self, X_train, X_test, y_train, y_test, labelList):
        if self.initialized == True:
            print("[INFO] Fitting and testing the model.")

            startTime = time()

            self.model.fit(X_train, np.ravel(y_train))
            test_accuracy = self.model.predict(X_test)

            print(classification_report(y_test, test_accuracy, target_names=labelList, digits=4))
            print("[INFO] Confusion Matrix:")
            print(confusion_matrix(y_test, test_accuracy))

            endTime = time()
            elapsed = endTime - startTime
            print('[INFO] Time taken: %f seconds.' % elapsed)
        else:
            print("[ERROR] Cannot run improperly-initialized model.")

