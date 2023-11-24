import numpy as np
from scipy.stats import uniform
from meta.parameter_tuning.grid_search import perform_grid_search
from meta.parameter_tuning.random_search import perform_random_search

class EstimatorSearchWrapper:
    def __init__(self, cfg, query_start_train, query_start_test, queries_train, queries_test, cv, logging_level):
        self.cfg = cfg
        self.query_start_train = query_start_train
        self.query_start_test = query_start_test
        self.queries_train = queries_train
        self.queries_test = queries_test
        self.cv = cv
        self.logging_level = logging_level

    def perform_search(self, estimator_class, param_grid, search_type='grid', n_iter=100):
        estimator = estimator_class(self.cfg, self.query_start_train)

        if search_type == 'grid':
            search = perform_grid_search(estimator, param_grid, self.queries_train, cv=self.cv, verbose=self.logging_level)
        elif search_type == 'random':
            search = perform_random_search(estimator, param_grid, self.queries_train, cv=self.cv, verbose=self.logging_level, n_iter=n_iter)
        else:
            raise ValueError("Invalid search type. Choose 'grid' or 'random'.")

        best_params = search.best_params_
        best_model = estimator_class(self.cfg, self.query_start_test, **best_params)
        test_score = best_model.score(self.queries_test)

        print("\nEstimator: " + estimator.__class__.__name__)
        print("Best parameters(" + search_type + "_search): " + str(best_params))
        print("Training score(" + search_type + "_search): " + str(search.best_score_))
        print("Test score (" + search_type + "_search): "  + str(test_score))

        return best_model, test_score