import math
import sys
import time
import metapy
import pytoml
import numpy as np
from meta.parameter_tuning.okapi_ndcg_estimator import OkapiBM25NdcgEstimator
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)


    # Define the parameter grid
    param_grid = {
        'k1': np.linspace(1.6, 1.8, num=20),
        'b': np.linspace(0.7, 0.8, num=20),
        #'k3': np.linspace(400, 600, num=20)
    }

    # Initialize the estimator
    estimator = OkapiBM25NdcgEstimator(cfg, query_start)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    grid_search = GridSearchCV(estimator, param_grid, cv=3, verbose=3)
    # Fit and score the estimator
    # X would be your queries and y would be the true relevance scores
    with open(query_path) as query_file:
        queries = [line.strip() for line in query_file]
    grid_search.fit(queries)

    print('Best parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

