"""
The SCORE operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import sys

from qryeval_plus.core.Idx import Idx
from qryeval_plus.query.QrySop import QrySop
from qryeval_plus.retrieval.RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from qryeval_plus.retrieval.RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from qryeval_plus.retrieval.RetrievalModelBM25 import RetrievalModelBM25

from qryeval_plus.query.QryIop import QryIop


class QrySopScore(QrySop):
    """
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.
        r: The retrieval model that determines what is a match.

        Returns True if the query matches, otherwise False.
        """
        return(self.docIteratorHasMatchFirst(r))


    def getScore(self, r):
        """
        Get a score for the document that docIteratorHasMatch matched.
        
        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """

        if isinstance(r, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnrankedBoolean(r)
        elif isinstance(r, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(r)
        elif isinstance(r, RetrievalModelBM25):
            return self.__getScoreBM25(r)
        else:
            raise Exception(
                '{} does not support the #SCORE operator.'.format(
                    r.__class__.__name__))


    def __getScoreUnrankedBoolean(self, r):
        """
        getScore for the Unranked retrieval model.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        else:
            return 1.0
        
    def __getScoreRankedBoolean(self, r):
        """
        Ranked Boolean score: tf of the term in the matched document
        """
        if not self.docIteratorHasMatchCache():
            return 0.0

        q = self._args[0]      # This is a QryIopXxx
        posting = q.docIteratorGetMatchPosting()
        return float(posting.tf)
    
    def __getScoreBM25(self, r):
        """
        BM25 score: computed using the BM25 formula
        """
        if not self.docIteratorHasMatchCache():
            return 0.0

        q = self._args[0]      # This is a QryIopXxx
        posting = q.docIteratorGetMatchPosting()
        field = q._field
        
        # Only fetch document-specific data here
        tf = posting.tf
        docLen = Idx.getFieldLength(field, posting.docid)

        # Use pre-calculated avgDocLen and idf from initialize()
        if self.avgDocLen == 0:
            K = r.k_1
        else:
            K = r.k_1 * ((1 - r.b) + r.b * (docLen / self.avgDocLen))
            
        tf_weight = ((tf) / (tf + K))

        score = self.idf * tf_weight
        return score

    def initialize(self, r):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators.  If the query operator is of type QryIop, it
        is fully evaluated, and the results are stored in an internal
        inverted list that may be accessed via the internal iterator.
        """
        q = self._args[ 0 ]
        q.initialize(r)

        # --- CACHING BM25 CONSTANTS ---
        # Calculate corpus-level stats once per query node instead of per document
        if isinstance(r, RetrievalModelBM25):
            field = q._field
            N = Idx.getNumDocs()
            df = q.getDf()
            
            sum_lengths = Idx.getSumOfFieldLengths(field)
            doc_count = Idx.getDocCount(field)
            
            if doc_count > 0:
                self.avgDocLen = sum_lengths / doc_count
            else:
                self.avgDocLen = 0.0
                
            # Pre-calculate IDF
            self.idf = math.log((N + 1) / (df + 0.5))