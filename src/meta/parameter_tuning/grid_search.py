from sklearn.model_selection import GridSearchCV

def perform_grid_search(estimator, param_grid, queries, cv=5, verbose=0):

    # Initialize the grid search
    # cv is the number of folds for cross validation
    grid_search = GridSearchCV(estimator, param_grid, cv=cv, verbose=verbose)
    
    # Fit and score the estimator
    grid_search.fit(queries)

    return grid_search
