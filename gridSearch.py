from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def gridSearch(X, y, cv, jobs):
    # Hyperparameter Tuning (Using 256 samples for each label)
    # Best accuracy for raw image: 0.7191570881226054
    # Parameters: {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'}

    grid_params = { 'n_neighbors' : [5,7,9,11,13,15,17,19,21],
                'weights' : ['uniform','distance'],
                'metric' : ['minkowski','euclidean','manhattan']}


    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    g_res = gs.fit(X, y)
    print(g_res.best_score_)
    print(g_res.best_params_)

