"""
Create, access, and manipulate inverted lists.  This inverted
list datatype is intended to be simpler to use than Lucene's.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from Idx import Idx
import PyLu


class InvList:
    """
    Create, access, and manipulate inverted lists.  This inverted
    list datatype is intended to be simpler to use than Lucene's.

    Attributes:
    
    ctf:
        Collection term frequency. The total number of term occurrences
        (positions) in the inverted list.

    df:
        Document frequency. The number of documents in the inverted list.

    postings:
        A list of DocPosting objects (see below).
    """

    # ----------------- Nested classes --------------------- #

    class DocPosting:
        """
        Utility class that makes it easier to construct postings.

        docid: An internal document id (an integer).
        locations: A list of document locations.

        Attributes:

        docid:
            An internal document id.

        tf:
            Term frequency of the term in the document. Also, the length
            of the positions list.

        positions:
            A list of locations in the document where the term occurs.
        """

        def __init__(self, d, locations):
            self.docid = d
            self.tf = len(locations)
            self.positions = locations


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, fieldString, termString=None):
        """
        If no TermString is provided, return an empty inverted list.
        Otherwise return an inverted list from the index.

        fieldString: The name of a document field.
        termString: A lexically-processed term that may be in the corpus.
        """

        # Object initialization
        self._field = fieldString
        self.ctf = 0
        self.df = 0
        self.postings = []

        if termString is None:
            return

        # Prepare to access the index.
        termLBytes = PyLu.LBytesRef(PyLu.JString(termString))
        term = PyLu.LTerm(PyLu.JString(fieldString), termLBytes)

        if Idx.indexReader.docFreq(term) < 1:
            return

        # Lucene indexes have segments, so postings must be retrieved
        # from each segment.  Some segments may have no postings.
        for context in Idx.indexReader.leaves():

            postings = context.reader().postings(term,
                                                  PyLu.LPostingsEnum.POSITIONS)

            if postings != None:

                # Convert from Lucene inverted list format to our inverted
                # list format. This is a little inefficient, but allows query
                # operators such as #SYN and #NEAR/n to be insulated from the
                # details of Lucene inverted list implementations.

                while postings.nextDoc() != PyLu.LDocIdSetIterator.NO_MORE_DOCS:
                    docid = context.docBase + postings.docID()
                    tf = postings.freq()
                    positions = [ None ] * tf

                    for j in range(0, tf):
                        positions[j] = postings.nextPosition();

                    posting = InvList.DocPosting(docid, positions) 
                    self.postings.append(posting)
                    self.df += 1
                    self.ctf += tf


    def __str__(self):
        """Render the inverted list as a string. Old InspectIndex format."""
   
        s = '\tdf:  {}\n\tctf:  {}\n'.format(self.df, self.ctf)

        for i in range(0, self.df):
            s += '\tdocid: {}\n\ttf: {}\n\tPositions: '.format(
                self.postings[i].docid,
                self.postings[i].tf)
            for j in range(0, len(self.postings[i].positions)):
                s += '{} '.format(self.postings[i].positions[j])
            s += '\n'

        return s


    def __str__new(self):
        """Render the inverted list as a string."""
   
        s = 'df: {}, ctf: {}\n'.format(self.df, self.ctf)

        for i in range(0, self.df):
            s += 'docid: {}, tf: {}, locs: {}\n'.format(
                self.postings[i].docid,
                self.postings[i].tf,
                str(self.postings[i].positions))

        return s


    def appendPosting(self, docid, positions):
        """
        Append a posting to the posting list.  Postings must be appended
        in docid order, otherwise this method fails.

        docid: An internal document id (an integer).
        positions: A list of document locations where the term occurs.

        Returns True if the posting was added, otherwise False.
        """

        # A posting can only be appended if its docid is greater than
        # the last docid.

        if ((self.df > 1) and
            (self.postings[self.df-1].docid >= docid)):
            return False

        posting = InvList.DocPosting(docid, positions)
        self.postings.append(posting)
        self.df += 1
        self.ctf += posting.tf
        return True


    def getDocid(self, n):
        """
        Get the n'th document id from the inverted list.

        n: An integer from 0 to df-1 that indicates which document
           in the inverted list.

        Returns the internal docid of the n'th document.
        """
        return(self.postings[n].docid)


    def getTf(self, n):
        """
        Get the term frequency in the n'th document of the inverted list.

        n: An integer from 0 to df-1 that indicates which document
           in the inverted list.

        Returns the term frequency in the n'th document.
        """
        return(self.postings[n].tf)

