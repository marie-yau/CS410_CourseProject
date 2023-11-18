from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class AbsoluteDiscountNdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, delta=0.7):
        self.delta = delta
        super().__init__(cfg, query_start, top_k)

    def set_ranker(self):
        return metapy.index.AbsoluteDiscount(self.delta)
