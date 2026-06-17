"""
The TERM operator for all retrieval models.  The TERM operator
stores information about a query term, for example "apple" in the
query "#AND (apple pie).  Although it may seem odd to use a query
operator to store a term, doing so makes it easy to build structured
queries with nested query operators.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from InvList import InvList
from QryIop import QryIop

class QryIopTerm(QryIop):
    """
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, termString, fieldString="body"):
        """
        Create a query node for a term that matches in a specified field.

        termString: A term string.
        fieldString: An optional field string. The default is the body field.
        """

        QryIop.__init__(self)		# Inherit from QryIop
        self._term = termString
        self._field = fieldString


    def __str__(self):
        """
        Get a string version of this query operator.
        Returns the string version of this query operator.
        """
        return(self._term + '.' + self._field)


    def evaluate(self):
        """
        Evaluate the term. The result is an internal inverted
        list that may be accessed via the internal iterators.

        @throws IOException: Error accessing the Lucene index.
        """
        self.invertedList = InvList(self._field, self._term)


