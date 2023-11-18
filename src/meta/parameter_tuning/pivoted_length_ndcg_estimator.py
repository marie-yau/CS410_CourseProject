from meta.parameter_tuning.ndcg_estimator import NdcgEstimator
import metapy

class PivotedLengthNdcgEstimator(NdcgEstimator):
    def __init__(self, cfg, query_start, top_k=10, s=0.2):
        self.s = s
        super().__init__(cfg, query_start, top_k)

    def set_ranker(self):
        return metapy.index.PivotedLength(self.s)
