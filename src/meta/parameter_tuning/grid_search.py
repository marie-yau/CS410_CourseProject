from sklearn.model_selection import GridSearchCV

def perform_grid_search(estimator, param_grid, query_path, cv=5, verbose=0):

    # Initialize the grid search
    # cv is the number of folds for cross validation
    grid_search = GridSearchCV(estimator, param_grid, cv=cv, verbose=verbose)
    
    # Fit and score the estimator
    with open(query_path) as query_file:
        queries = [line.strip() for line in query_file]
    grid_search.fit(queries)

    return grid_search
