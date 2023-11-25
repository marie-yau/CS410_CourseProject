import sys
import pytoml
import numpy as np
import matplotlib.pyplot as plt
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


ESTIMATOR_TO_PARAM_GRID = {
    OkapiBM25NdcgEstimator:  {'k1': np.linspace(1.6, 1.8, num=20), 'b': np.linspace(0.7, 0.8, num=20)},
    PivotedLengthNdcgEstimator: {'s': np.linspace(0, 8, num=200)},
    AbsoluteDiscountNdcgEstimator: {'delta': np.linspace(0, 8, num=200)},
    JelinekMercerNdcgEstimator: { 'l': np.linspace(0, 8, num=200)},
    DirichletPriorNdcgEstimator: {'mu': np.linspace(0, 5000, num=200)}
}

SEARCH_TYPES = ['grid', 'random']


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


    test_scores = []
    model_identifiers = []

    overall_best_estimator = None
    overall_best_model = None
    overall_best_test_score = 0
    for estimator in ESTIMATOR_TO_PARAM_GRID:
        param_grid = ESTIMATOR_TO_PARAM_GRID[estimator]
        for search_type in SEARCH_TYPES:
            best_model, test_score = wrapper.perform_search(estimator, param_grid, search_type=search_type)
            test_scores.append(test_score)
            model_identifiers.append(estimator.__name__ + "-" + search_type)
            if test_score > overall_best_test_score:
                overall_best_estimator = estimator
                overall_best_model = best_model
                overall_best_test_score = test_score

    print("Overall best model: " + str(overall_best_model))
    print("Best Test score: "  + str(overall_best_test_score))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_identifiers, test_scores, color='skyblue')
    plt.xlabel('Model Identifier')
    plt.ylabel('Test Score')
    plt.title('Test Scores for Each Iteration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results.png')

