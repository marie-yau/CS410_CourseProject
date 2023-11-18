from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class OkapiBM25NdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, k1=1.2, b=0.75, k3=500.0):
        self.k1 = k1
        self.b = b
        self.k3 = k3
        super().__init__(cfg, query_start, top_k)
    
    def set_ranker(self):
        return metapy.index.OkapiBM25(self.k1, self.b, self.k3)
