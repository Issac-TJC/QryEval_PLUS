"""
The AND operator for Unranked Boolean and Ranked Boolean retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from qryeval_plus.query.QrySop import QrySop
from qryeval_plus.retrieval.RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from qryeval_plus.retrieval.RetrievalModelRankedBoolean import RetrievalModelRankedBoolean


class QrySopAnd(QrySop):
    """
    The AND operator for Boolean retrieval models.
    """

    def __init__(self):
        QrySop.__init__(self)


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.

        AND: a document matches iff ALL arguments match the same document.
        """
        return self.docIteratorHasMatchAll(r)


    def getScore(self, retrievalModel):
        """
        Get a score for the document that docIteratorHasMatch matched.
        """

        if isinstance(retrievalModel, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnrankedBoolean(retrievalModel)
        elif isinstance(retrievalModel, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(retrievalModel)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                retrievalModel.__class__.__name__))

    def __getScoreUnrankedBoolean(self, r):
        """
        getScore for Unranked Boolean retrieval model.

        AND: If all arguments match, return 1.0.
        """

        if self.docIteratorHasMatchCache():
            return 1.0
        return 0.0

    def __getScoreRankedBoolean(self, r):
        """
        getScore for Ranked Boolean retrieval model.

        AND uses MIN to combine argument scores.
        """

        score = float('inf')
        docid = self.docIteratorGetMatch()

        for q_i in self._args:
            # AND has matched, so all args must match
            if q_i.docIteratorGetMatch() == docid:
                score = min(score, q_i.getScore(r))

        return score
