from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def gridSearch(X, y, cv, jobs):
    # Hyperparameter Tuning (Using 256 samples for each label)

    grid_params = { 'n_neighbors' : [5,7,9,11,13,15,17,19,21],
                'weights' : ['uniform','distance'],
                'metric' : ['minkowski','euclidean','manhattan']}

    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=cv, n_jobs = jobs)
    g_res = gs.fit(X, y)
    print(g_res.best_score_)
    print(g_res.best_params_)

