"""
All query operators that return inverted lists are subclasses of
the QryIop class.  This class has two main purposes.  First, it
allows query operators to easily recognize any nested query
operator that returns an inverted list (e.g., #AND (a #NEAR/1 (b c)).
Second, it is a place to store data structures and methods that are
common to all query operators that return inverted lists.

After a QryIop operator is initialized, it caches a full inverted
list, and information from the inverted list is accessible.  Document
and location information are accessed via Qry.docIterator and
QryIop.locIterator.  Corpus-level information, for example, 
document frequency (df) and collection term frequency (ctf), are
available via specific methods (e.g., getDf and getCtf).

QryIop operators support iteration over the locations in the
document that Qry.docIteratorHasMatch matches.  The semantics
and use of the QryIop.locIterator are similar to the Qry.docIterator.
The QryIop.locIterator is initialized automatically each time
Qry.docIteratorHasMatch finds a match; no additional initialization
is required.

IMPLEMENTATION NOTES:
Iteration in QryIop and QrySop is very different.  In QryIop,
docIterator and locIterator iterate over the cached inverted
list, NOT recursively over the query arguments.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from Qry import Qry

class QryIop(Qry):
    """
    The root of the class hierarchy for inverted list operators.
    """

    # -------------- Constants and variables --------------- #

    INVALID_ITERATOR_INDEX = -1

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):

        Qry.__init__(self)		# Inherit from Qry
        self._field = None
        self._invertedList = None
        self._docIteratorIndex = QryIop.INVALID_ITERATOR_INDEX
        self._locIteratorIndex = QryIop.INVALID_ITERATOR_INDEX


    def docIteratorAdvancePast(self, docid):
        """
        Advance the query operator's internal iterator beyond the
        specified document.

        docid: The document's internal document id.
        """
        while ((self.docIteratorIndex < self.invertedList.df) and
               (self.invertedList.getDocid(self.docIteratorIndex) <=
                 docid)):
            self.docIteratorIndex += 1
           
        self.locIteratorIndex = 0


    def docIteratorAdvanceTo(self, docid):
        """
        Advance the query operator's internal iterator to the specified
        document if it exists, or beyond if it doesn't.

        docid: The document's internal document id.
        """
        while ((self.docIteratorIndex < self.invertedList.df) and
               (self.invertedList.getDocid(self.docIteratorIndex) <
                 docid)):
            self.docIteratorIndex += 1

        self.locIteratorIndex = 0


    def docIteratorFinish(self):
        """
        Advance the query operator's internal iterator beyond the
        any possible document.
        """
        self.docIteratorIndex = len(self.invertedList.postings)


    def docIteratorGetMatch(self):
        """
        Return the id of the document that the query operator's internal
        iterator points to now.  Use docIteratorHasMatch to determine whether
        the iterator currently points to a document.  If the iterator
        doesn't point to a document, an invalid document id is returned.

        Returns the internal id of the current document.
        """
        return(self.invertedList.getDocid(self.docIteratorIndex))


    def docIteratorGetMatchPosting(self):
        """
        Return the postings for the document that the docIterator points to
        now, or throw an error if the docIterator doesn't point at a document.
        Returns a document posting.
        """
        return(self.invertedList.postings[self.docIteratorIndex])


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a matching document.

        r: A retrieval model (that is ignored - it can be null)
        Returns True if the query matches a document, otherwise False.
        """
        return(self.docIteratorIndex < self.invertedList.df)


    def getCtf(self):
        """
        Get the collection term frequency (ctf) associated with this
        query operator.  It is an error to call this method before the
        object's initialize method is called.

        Returns the collection term frequency (ctf).
        """
        return(self.invertedList.ctf)


    def getDf(self):
        """
        Get the document frequency (df) associated with this query
        operator.  It is an error to call this method before the
        object's initialize method is called.

        Returns the document frequency (df).
        """
        return(self.invertedList.df)


    def initialize(self, r):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators; this method must be called before iteration
        can begin.

        r: A retrieval model (that is ignored)
        """

        # Initialize the query arguments (if any).
        for q_i in self._args:
            q_i.initialize(r)

        # Evaluate the operator.
        self.evaluate()

        # Initialize the internal iterators.
        self.docIteratorIndex = 0
        self.locIteratorIndex = 0


    def locIteratorAdvance(self):
        """
        Advance the query operator's internal iterator to the
        next location.
        """
        self.locIteratorIndex += 1


    def locIteratorAdvancePast(self, loc):
        """
        Advance the query operator's internal iterator beyond the
        specified location.
        loc: The location to advance beyond.
        """
        tf = self.invertedList.postings[self.docIteratorIndex].tf
        positions = self.invertedList.postings[self.docIteratorIndex].positions

        while (self.locIteratorIndex < tf and
                positions[self.locIteratorIndex] <= loc):
            self.locIteratorIndex += 1


    def locIteratorAdvanceTo(self, loc):
        """
        Advance the query operator's internal iterator to the specified
        location if it exists, or beyond if it doesn't.

        loc: The location to advance to or beyond.
        """
        tf = self.invertedList.postings[self.docIteratorIndex].tf
        positions = self.invertedList.postings[self.docIteratorIndex].positions

        while (self.locIteratorIndex < tf and
                positions[self.locIteratorIndex] < loc):
            self.locIteratorIndex += 1


    def locIteratorFinish(self):
        """
        Advance the query operator's internal iterator beyond
        any possible location.
        """
        self.locIteratorIndex = self.invertedList.postings[self.docIteratorIndex].tf


    def locIteratorGetMatch(self):
        """
        Return the document location that the query operator's internal
        iterator points to now.  Use locIteratorHasMatch to determine
        whether the iterator currently points to a location.  If the
        iterator doesn't point to a location, an invalid document
        location is returned.
        """
        locations = self.docIteratorGetMatchPosting().positions
        return(locations[self.locIteratorIndex])


    def locIteratorHasMatch(self):
        """
        Returns true if the query operator's internal iterator currently
        points to a location.
        Returns True if the iterator currently points to a location.
        """
        return(self.locIteratorIndex <
                self.invertedList.getTf(self.docIteratorIndex))
