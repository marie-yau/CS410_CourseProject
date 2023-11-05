from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class NdcgEstimator(BaseEstimator, ABC):
    def __init__(self, cfg, query_start, top_k=10):
        self.ranker = None
        self.cfg = cfg
        self.top_k = top_k
        self.query_start = query_start

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        ndcg = 0.0
        num_queries = 0
        for query_num, line in enumerate(X):
            self.query.content(line)
            results = self.ranker.score(self.idx, self.query, self.top_k)
            ndcg += self.ev.ndcg(results, self.query_start + query_num, self.top_k)
            num_queries+=1
        return ndcg / num_queries
