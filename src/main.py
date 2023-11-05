import sys
import pytoml
import numpy as np
from meta.parameter_tuning.okapi_ndcg_estimator import OkapiBM25NdcgEstimator
from meta.parameter_tuning.grid_search import perform_grid_search

def process_cfg(cfg):
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    return query_path, query_start

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    query_path, query_start = process_cfg(cfg)


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
    grid_search = perform_grid_search(estimator, param_grid, query_path, cv=3, verbose=3)

    print('Best parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

