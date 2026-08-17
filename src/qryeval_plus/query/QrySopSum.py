"""
The SUM operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from qryeval_plus.query.QrySop import QrySop
from qryeval_plus.retrieval.RetrievalModelBM25 import RetrievalModelBM25


class QrySopSum(QrySop):
    """
    The SUM  operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__( self ):
        QrySop.__init__( self )		# Inherit from QrySop


    # STUDENTS:
    # Add new methods below. See QrySop.py for guidance about
    # the new methods that you need to define.
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
        """
        SUM = sum(score_i)
        """
        score = 0.0
        docid = self.docIteratorGetMatch()

        for q in self._args:
            if q.docIteratorHasMatch(r) and q.docIteratorGetMatch() == docid:
                score += q.getScore(r)

        return score