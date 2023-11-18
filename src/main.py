import sys
import pytoml
import numpy as np
from scipy.stats import uniform
from meta.parameter_tuning.okapi_ndcg_estimator import OkapiBM25NdcgEstimator
from meta.parameter_tuning.pivoted_length_ndcg_estimator import PivotedLengthNdcgEstimator
from meta.parameter_tuning.absolute_discount_ndcg_estimator import AbsoluteDiscountNdcgEstimator
from meta.parameter_tuning.jelinek_mercer_ndcg_estimator import JelinekMercerNdcgEstimator
from meta.parameter_tuning.dirichlet_prior_ndcg_estimator import DirichletPriorNdcgEstimator
from meta.parameter_tuning.grid_search import perform_grid_search
from meta.parameter_tuning.random_search import perform_random_search

def process_query_cfg(cfg):
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d.get('query-runner')
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    return query_path, query_start

def process_param_tuner_cfg(cfg):
    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    tuner_cfg = cfg_d.get('params-tuner')
    if tuner_cfg is None:
        print("params_tuner table needed in {}".format(cfg))
        sys.exit(1)

    logging_level = tuner_cfg.get('logging-level', 0)
    split_ratio = tuner_cfg.get('split-ratio', 0.8)
    cv = tuner_cfg.get('cv', 5)

    return logging_level, split_ratio, cv

def train_test_split(cfg, query_path, query_start, split_ratio):
    # Fit and score the estimator
    with open(query_path) as query_file:
        queries = [line.strip() for line in query_file]
        split_index = int(len(queries) * split_ratio)
        queries_train = queries[:split_index]
        queries_test = queries[split_index:]
    return queries_train, queries_test, query_start, query_start + split_index


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    query_path, query_start = process_query_cfg(cfg)
    logging_level, split_ratio, cv = process_param_tuner_cfg(cfg)

    
    # Split train/test data for prediction evaluation
    queries_train, queries_test, query_start_train, query_start_test = train_test_split(
        cfg, query_path, query_start, split_ratio
    )

    # Define the parameter grid
    param_grid = {
        'k1': np.linspace(1.6, 1.8, num=20),
        'b': np.linspace(0.7, 0.8, num=20),
        #'k3': np.linspace(400, 600, num=20)
    }

    # Initialize the estimator
    estimator = OkapiBM25NdcgEstimator(cfg, query_start_train)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    grid_search = perform_grid_search(estimator, param_grid, queries_train, cv=cv, verbose=logging_level)
    best_model = OkapiBM25NdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        k1 = grid_search.best_params_['k1'],
        b = grid_search.best_params_['b'],
        # k3 = grid_search.best_params_['k3']
    )
    test_score = best_model.score(queries_test)

    print('Best parameters (grid_search): ', grid_search.best_params_)
    print('Training score (grid_search): ', grid_search.best_score_)
    print('Test score (grid_search): ', test_score)


    param_distributions = {
        'k1': uniform(loc=1.6, scale=0.2),  # uniform distribution from 1.6 to 1.8
        'b': uniform(loc=0.7, scale=0.1),  # uniform distribution from 0.7 to 0.8
    }
    random_search = perform_random_search(
        estimator, param_distributions, queries_train, cv=cv, verbose=logging_level, n_iter=100
    )
    best_model = OkapiBM25NdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        k1 = random_search.best_params_['k1'],
        b = random_search.best_params_['b'],
        # k3 = random_search.best_params_['k3']
    )
    test_score = best_model.score(queries_test)

    print('Best parameters (random_search): ', random_search.best_params_)
    print('Training score (random_search): ', random_search.best_score_)
    print('Test score (random_search): ', test_score)
    
    # Define the parameter grid
    pl_param_grid = {
        's': np.linspace(0, 8, num=200),
    }

    # Initialize the estimator
    pl_estimator = PivotedLengthNdcgEstimator(cfg, query_start_train)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    pl_grid_search = perform_grid_search(pl_estimator, pl_param_grid, queries_train, cv=cv, verbose=logging_level)
    pl_best_model = PivotedLengthNdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        s = pl_grid_search.best_params_['s'],
    )
    pl_test_score = pl_best_model.score(queries_test)

    print('PivotedLength Best parameters (grid_search): ', pl_grid_search.best_params_)
    print('PivotedLength Training score (grid_search): ', pl_grid_search.best_score_)
    print('PivotedLength Test score (grid_search): ', pl_test_score)

    # Define the parameter grid
    ad_param_grid = {
        'delta': np.linspace(0, 8, num=200),
    }

    # Initialize the estimator
    ad_estimator = AbsoluteDiscountNdcgEstimator(cfg, query_start_train)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    ad_grid_search = perform_grid_search(ad_estimator, ad_param_grid, queries_train, cv=cv, verbose=logging_level)
    ad_best_model = AbsoluteDiscountNdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        delta = ad_grid_search.best_params_['delta'],
    )
    ad_test_score = ad_best_model.score(queries_test)

    print('AbsoluteDiscount Best parameters (grid_search): ', ad_grid_search.best_params_)
    print('AbsoluteDiscount Training score (grid_search): ', ad_grid_search.best_score_)
    print('AbsoluteDiscount Test score (grid_search): ', ad_test_score)

    # Define the parameter grid
    jm_param_grid = {
        'l': np.linspace(0, 8, num=200),
    }

    # Initialize the estimator
    jm_estimator = JelinekMercerNdcgEstimator(cfg, query_start_train)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    jm_grid_search = perform_grid_search(jm_estimator, jm_param_grid, queries_train, cv=cv, verbose=logging_level)
    jm_best_model = JelinekMercerNdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        l = jm_grid_search.best_params_['l'],
    )
    jm_test_score = jm_best_model.score(queries_test)

    print('JelinekMercer Best parameters (grid_search): ', jm_grid_search.best_params_)
    print('JelinekMercer Training score (grid_search): ', jm_grid_search.best_score_)
    print('JelinekMercer Test score (grid_search): ', jm_test_score)

    # Define the parameter grid
    dp_param_grid = {
        'mu': np.linspace(0, 5000, num=200),
    }

    # Initialize the estimator
    dp_estimator = DirichletPriorNdcgEstimator(cfg, query_start_train)

    # Initialize the grid search
    # cv is the number of folds for cross validation
    dp_grid_search = perform_grid_search(dp_estimator, dp_param_grid, queries_train, cv=cv, verbose=logging_level)
    dp_best_model = DirichletPriorNdcgEstimator(
        cfg = cfg, 
        query_start = query_start_test,
        mu = dp_grid_search.best_params_['mu'],
    )
    dp_test_score = dp_best_model.score(queries_test)

    print('DirichletPrior Best parameters (grid_search): ', dp_grid_search.best_params_)
    print('DirichletPrior Training score (grid_search): ', dp_grid_search.best_score_)
    print('DirichletPrior Test score (grid_search): ', dp_test_score)
