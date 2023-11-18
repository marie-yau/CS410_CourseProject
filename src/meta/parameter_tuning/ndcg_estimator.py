from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import metapy

class NdcgEstimator(BaseEstimator, ABC):
    def __init__(self, cfg, query_start, top_k=10):
        self.cfg = cfg
        self.top_k = top_k
        self.query_start = query_start
        self._metapy_init()
        self.ranker = self.set_ranker()

    def fit(self, X, y=None):
        self._metapy_init()
        self.ranker = self.set_ranker()
        return self

    def score(self, X, y=None):
        ndcg = 0.0
        num_queries = 0
        for query_num, line in enumerate(X):
            self.query.content(line)
            results = self.ranker.score(self.idx, self.query, self.top_k)
            ndcg += self.ev.ndcg(results, self.query_start + query_num, self.top_k)
            num_queries+=1
        return ndcg / num_queries
    
    def _metapy_init(self):
        self.idx = metapy.index.make_inverted_index(self.cfg)
        self.ev = metapy.index.IREval(self.cfg)
        self.query = metapy.index.Document()

    @abstractmethod
    def set_ranker(self):
        """Set the ranking function"""
