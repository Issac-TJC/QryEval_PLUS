"""
The WSUM operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from QrySop import QrySop
from RetrievalModelBM25 import RetrievalModelBM25


class QrySopWsum(QrySop):
    """
    The WSUM operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__( self ):
        QrySop.__init__( self )		# Inherit from QrySop

        self._weights = []
        self._weight_sum = 0.0
        
    def appendArg(self, q, weight=1.0):
        QrySop.appendArg(self, q)
        
        self._weights.append(float(weight))
        self._weight_sum += float(weight)

    def docIteratorHasMatch(self, r):
        """
        A document matches if at least one argument matches.
        """
        return self.docIteratorHasMatchMin(r)
    
    def getScore(self, r):
        """
        Get the weighted sum score for the current matching document.
        """
        if isinstance(r, RetrievalModelBM25):
            return self.__getScoreBM25(r)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                r.__class__.__name__))

    def __getScoreBM25(self, r):
        score = 0.0
        docid = self.docIteratorGetMatch()

        for i, q in enumerate(self._args):
            if q.docIteratorHasMatch(r) and q.docIteratorGetMatch() == docid:
                score += self._weights[i] * q.getScore(r)

        return score / self._weight_sum if self._weight_sum > 0 else 0.0