"""
QrySop is the root class of all query operators that use a
retrieval model to determine whether a query matches a document and to
calculate a score for the document.  This class has two main purposes.
First, it allows query operators to easily recognize any nested query
operator that returns a document scores (e.g., #AND (a #OR(b c)).
Second, it is a place to store data structures and methods that are
common to all query operators that calculate document scores.

"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from Qry import Qry

class QrySop(Qry):
    """
    The root of the class hierarchy for score list operators.
    """

    # -------------- Constants and variables --------------- #



    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        Qry.__init__(self)		# Inherit from Qry


    def getDefaultScore(retrievalModel,  docid):
        """
        Get a score that indicates how well the query matches the specified
        document (which is assumed to not contain the query term(s)).

        retrievalModel: retrieval model parameters
        docid: an internal document id

        Returns a score that indicates how well the query matches the document.
        
        throws IOException: Error accessing the Lucene index
        """
        raise Exception('Each QrySop method needs to implement ' +
                         sys._getframe().f_code.co_name)


    def getScore(self, retrievalModel):
        """
        Get a score for the document that docIteratorHasMatch matched.

        retrievalModel: retrieval model parameters

        Returns the document score.

        throws IOException: Error accessing the Lucene index
        """
        raise Exception('Each QrySop method needs to implement ' +
                         sys._getframe().f_code.co_name)


    def initialize(self, retrievalModel):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators.  If the query operator is of type QryIop, it
        is fully evaluated, and the results are stored in an internal
        inverted list that may be accessed via the internal iterator.

        retrievalModel: A retrieval model that guides initialization

        throws IOException: Error accessing the Lucene index.
        """
        for q_i in self._args:
            q_i.initialize(retrievalModel)
