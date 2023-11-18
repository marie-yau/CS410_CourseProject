from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class DirichletPriorNdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, mu=2000.0):
        self.mu = mu
        super().__init__(cfg, query_start, top_k)

    def set_ranker(self):
        return metapy.index.DirichletPrior(self.mu)
