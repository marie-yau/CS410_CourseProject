from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class OkapiBM25NdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, k1=1.2, b=0.75, k3=500.0):
        super().__init__(cfg, query_start, top_k)
        self.k1 = k1
        self.b = b
        self.k3 = k3

    def fit(self, X, y=None):
        self.ranker = metapy.index.OkapiBM25(self.k1, self.b, self.k3)
        self.idx = metapy.index.make_inverted_index(self.cfg)
        self.ev = metapy.index.IREval(self.cfg)
        self.query = metapy.index.Document() 
        return self
