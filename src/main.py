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
from meta.estimator_search.estimator_search_wrapper import EstimatorSearchWrapper
from meta.utils.process_configs import process_query_cfg, process_param_tuner_cfg
from meta.utils.eval_utils import train_test_split



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

    wrapper = EstimatorSearchWrapper(cfg, query_start_train, query_start_test, queries_train, queries_test, cv, logging_level)

    # OkapiBM25NdcgEstimator
    okapi_param_grid = {'k1': np.linspace(1.6, 1.8, num=20), 'b': np.linspace(0.7, 0.8, num=20)}
    wrapper.perform_search(OkapiBM25NdcgEstimator, okapi_param_grid, search_type='grid')
    wrapper.perform_search(OkapiBM25NdcgEstimator, okapi_param_grid, search_type='random')

    # PivotedLengthNdcgEstimator
    pl_param_grid = {'s': np.linspace(0, 8, num=200)}
    wrapper.perform_search(PivotedLengthNdcgEstimator, pl_param_grid, search_type='grid')
    wrapper.perform_search(PivotedLengthNdcgEstimator, pl_param_grid, search_type='random')

    # AbsoluteDiscountNdcgEstimator
    ad_param_grid = {'delta': np.linspace(0, 8, num=200)}
    wrapper.perform_search(AbsoluteDiscountNdcgEstimator, ad_param_grid, search_type='grid')
    wrapper.perform_search(AbsoluteDiscountNdcgEstimator, ad_param_grid, search_type='random')

    # # Define the parameter grid
    jm_param_grid = { 'l': np.linspace(0, 8, num=200)}
    wrapper.perform_search(JelinekMercerNdcgEstimator, jm_param_grid, search_type='grid')
    wrapper.perform_search(JelinekMercerNdcgEstimator, jm_param_grid, search_type='random')

    dp_param_grid = {'mu': np.linspace(0, 5000, num=200)}
    wrapper.perform_search(DirichletPriorNdcgEstimator, dp_param_grid, search_type='grid')
    wrapper.perform_search(DirichletPriorNdcgEstimator, dp_param_grid, search_type='random')
