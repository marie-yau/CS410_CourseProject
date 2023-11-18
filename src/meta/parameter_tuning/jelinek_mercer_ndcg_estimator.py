from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class JelinekMercerNdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, l=0.7):
        self.l = l
        super().__init__(cfg, query_start, top_k)

    def set_ranker(self):
        return metapy.index.JelinekMercer(self.l)
