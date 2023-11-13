from sklearn.model_selection import RandomizedSearchCV

def perform_random_search(estimator, param_distributions, queries, cv=5, verbose=0, n_iter=100):

    # Initialize the random search
    # cv is the number of folds for cross validation
    # n_iter is the number of parameter settings that are sampled
    random_search = RandomizedSearchCV(estimator, param_distributions, cv=cv, verbose=verbose, n_iter=n_iter)
    
    # Fit and score the estimator
    random_search.fit(queries)

    return random_search
