from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time

def gridSearch(X, y, modelType, cv, jobs = -1):
    # Hyperparameter Tuning 
    startTime = time()
    initialized = False

    if modelType == "knn":
        initialized = True
        grid_params = {
            'n_neighbors' : [5,7,9,11,13,15],
            'weights' : ['uniform','distance'],
            'metric' : ['minkowski','euclidean','manhattan']
        }
        gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    elif modelType == "rf"  or modelType == "randomforest":
        initialized = True
        grid_params = {
            'bootstrap': [True],
            'max_depth': [50, 100, 200],
            'max_features': [2, 4],
            'min_samples_leaf': [3, 4],
            'min_samples_split': [6, 10],
            'n_estimators': [100, 200]
        }
        gs = GridSearchCV(RandomForestClassifier(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    elif modelType == "dt"  or modelType == "decisiontree":
        initialized = True
        grid_params = {
            'max_features': ['sqrt', 'log2'],
            'ccp_alpha': [0.1, .01, .001],
            'max_depth' : [5, 6, 7, 8, 9, 10,],
            'criterion' :['gini', 'entropy']
        }
        gs = GridSearchCV(DecisionTreeClassifier(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    elif modelType == "nb" or modelType == "naivebayes":
        print("[WARN] grid search tuning does not affect naive bayes.")
    elif modelType == "svm" or modelType == "svc":
        initialized = True
        grid_params = {
            'C': [0.5, 1, 1.5, 2], 
            'kernel': ['linear', 'rbf'], 
            'gamma': [0.5, 1, 1.5, 2]
        }
        gs = GridSearchCV(SVC(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    else:
        print("[ERROR] Invalid model name identifier detected.")
    if initialized == True:
        g_res = gs.fit(X, y)
        
        # calculate results and best parameters
        endTime = time()
        elapsed = endTime - startTime
        print('[INFO] Time taken: %f seconds.' % elapsed)
        print("[INFO] Best accuracy: ", g_res.best_score_)
        print("[INFO] Best parameters: ")
        print(g_res.best_params_)

