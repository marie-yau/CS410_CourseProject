from sklearn.model_selection import RandomizedSearchCV

def perform_random_search(estimator, param_distributions, query_path, cv=5, verbose=0, n_iter=100):

    # Initialize the random search
    # cv is the number of folds for cross validation
    # n_iter is the number of parameter settings that are sampled
    random_search = RandomizedSearchCV(estimator, param_distributions, cv=cv, verbose=verbose, n_iter=n_iter)
    
    # Fit and score the estimator
    with open(query_path) as query_file:
        queries = [line.strip() for line in query_file]
    random_search.fit(queries)

    return random_search
