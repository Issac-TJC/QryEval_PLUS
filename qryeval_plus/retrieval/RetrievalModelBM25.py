from qryeval_plus.retrieval.RetrievalModel import RetrievalModel

class RetrievalModelBM25(RetrievalModel):
    """
    Define and store data for the BM25 Retrieval Model.
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        RetrievalModel.__init__(self)		# Inherit from RetrievalModel
        self.k_1 = float(parameters.get('BM25:k_1', 1.2))
        self.b = float(parameters.get('BM25:b', 0.75))
        # self.k_3 = float(parameters.get('BM25:k_3', 8))
        self.defaultQrySop = '#SUM'