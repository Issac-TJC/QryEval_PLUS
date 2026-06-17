"""
The Qry class is the root class in the query operator hierarchy.
Most of this class is abstract, because different types of
query operators (Sop, Iop) have different subclasses, and each
query operator has its own subclass.  This class defines the
common interface to query operators, and is a place to store data
structures and methods that are common to all query operators.

Document-at-a-time (DAAT) processing is implemented as iteration
over (virtual or materialized) lists of document ids or document
locations.  To evaluate query q using the UnrankedBoolean retrieval
model:

  RetrievalModel r = new RetrievalModelUnrankedBoolean ();
  q.initialize (r);

  while (q.docIteratorHasMatch (r)) {
    int docid = q.docIteratorGetMatch ();
    double score = ((QrySop) q).getScore (model);
    System.out.println ("internal docid: " + docid + ", score: " score);
    q.docIteratorAdvancePast (docid);
  }

The Qry class defines the iteration interface and provides general
methods that each subclass may override or use.  Note that the 
iteration interface does not conform to the standard Python
iteration interface.  It has different characteristics and capabilities.
For example, getting the current element does not consume the
element; the iterator must be advanced explicitly, and it can be
advanced in different ways, which provides opportunities to evaluate
the query more efficiently.

The Qry class has two subclasses.  QrySop ("score operators") contains
query operators that compute document scores (e.g., AND, OR, SCORE).
QryIop ("inverted list operators") contains query operators that
produce inverted lists (e.g., SYN, NEAR, TERM).  

The docIterator for query operators in the QrySop hierarchy iterates
over a virtual list.  The next document id is determined dynamically
when hasMatch is called.  Thus, the iterator needs to be part of
the query operator, because different query operators may have
different strategies for determining what matches and how scores
are calculated.  When hasMatch identifies a match, the match is
cached so that it can be accessed efficiently by getMatch and
getScore methods.

The inverted lists of query operators in the QryIop hierarchy are
materialized when the query operator is initialized.  It is not
possible to produce them in a document-at-a-time mode because
the df and ctf statistics are not known until the inverted list
is fully constructed.  QryIop operators provide a document-at-a-time
interface to the inverted lists via docIterators.

The data structure that stores query arguments (args) is accessible
by subclasses.  If it is accessed via a standard Java iterator, the
search engine creates and then discards many (many) iterators during
query evaluation, which reduces computational efficiency.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

class Qry:
    """
    The root of the class hierarchy for all query operators.
    """

    # -------------- Constants and variables --------------- #

    INVALID_DOCID = -1

    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        self._args = []	# Arguments to this query operator.
        self._displayName = ''	# The name to display when rendering to string

        # docIteratorHasMatch caches the matching docid so that
        # docIteratorGetMatch and getScore don't have to recompute it.
        __docIteratorMatchCache = Qry.INVALID_DOCID
        __matchingDocid = None
        __matchStored = False


    def appendArg(self, q):
        """
        Append an argument to the list of query operator arguments.  

        q: The query argument (query operator) to append.

        throws: IllegalArgumentException q is an invalid argument
        """

        from QryIop import QryIop
        from QryIopTerm import QryIopTerm
        from QrySop import QrySop
        from QrySopScore import QrySopScore

        #  The query parser and query operator type system are too simple
        #  to detect some kinds of query syntax errors.  appendArg does
        #  additional syntax checking while creating the query tree.  It
        #  also inserts SCORE operators between QrySop operators and QryIop
        #  arguments, and propagates field information from QryIop
        #  children to parents.  Basically, it creates a well-formed
        #  query tree.
        if isinstance(self, QryIopTerm):
            raise Exception('The TERM operator has no arguments.')

        #  SCORE operators can have only a single argument of type QryIop.
        if isinstance(self, QrySopScore):
            if len(self._args) > 0:
                raise Exception('Score operators can have only one argument ')
            elif not isinstance(q, QryIop):
                raise Exception('The argument to a SCORE operator must be of type QryIop.')
            else:
                self._args.append(q)
                return

        #  Check whether it is necessary to insert an implied SCORE
        #  operator between a QrySop operator and a QryIop argument.
        if isinstance(self, QrySop) and isinstance(q, QryIop):
            impliedOp = QrySopScore()
            impliedOp.setDisplayName('#SCORE')
            impliedOp.appendArg(q)
            self._args.append(impliedOp)
            return

        #  QryIop operators must have QryIop arguments in the same field.
        if isinstance(self, QryIop) and isinstance(q, QryIop):
            if len(self._args) == 0:
                self._field = q._field
            else:
                if self._field != q._field:
                    raise Exception(
                        'Arguments to QryIop operators must be in the same field.')
            self._args.append(q)
            return

        #  QrySop operators and their arguments must be of the same type.
        if isinstance(self, QrySop) and isinstance(q, QrySop):
            self._args.append(q)
            return

        raise Exception(
            'Objects of type {} cannot be an argument to a query operator of type {}'.format(q.__class__.__name__, self.__class__.__name__))


    def docIteratorAdvancePast(self, docid):
        """
        Advance the internal document iterator beyond the specified
        document.

        docid: An internal document id.
        """
        for q_i in self._args:
            q_i.docIteratorAdvancePast(docid)

        self.docIteratorClearMatchCache()


    def docIteratorAdvanceTo(self, docid):
        """
        Advance the internal document iterator to the specified
        document, or beyond if it doesn't.

        docid: An internal document id.
        """
        for q_i in self._args:
            q_i.docIteratorAdvanceTo(docid)

        self.docIteratorClearMatchCache()


    def docIteratorClearMatchCache(self):
        """
        Clear the docIterator's matching docid cache.  The cache should
        be cleared whenever a docIterator is advanced.
        """
        self.docIteratorMatchCache = Qry.INVALID_DOCID


    def docIteratorGetMatch(self):
        """
        Return the id of the document that the iterator points to now.
        Use docIteratorHasMatch to determine whether the iterator
        currently points to a document.  If the iterator doesn't point
        to a document, an invalid document id is returned.

        Returns the internal id of the current document.
        """
        if self.docIteratorHasMatchCache():
            return self.docIteratorMatchCache
        else:
            raise Exception('No matching docid was cached.')


    def docIteratorHasMatch(self, r):
        """
        Indicate whether the query has a match.        
        r: The retrieval model that determines what is a match

        Returns True if the query matches, otherwise False.
        """
        raise Exception('{} needs to implement {}'.format(
            self.__class__.__name__,
            sys._getframe().f_code.co_name))


    def docIteratorHasMatchAll(self, r):
        """
        An instantiation of docIteratorHasMatch that is true if the
        query has a document that matches all query arguments; some
        subclasses may choose to use this implementation.  

        r: The retrieval model that determines what is a match.

        Returns True if the query matches, otherwise False.
        """

        matchFound = False

        # Keep trying until a match is found or no match is possible.
        while not matchFound:

            # Get the docid of the first query argument.
            q_0 = self._args[ 0 ]

            if not q_0.docIteratorHasMatch(r):
                return(False)

            docid_0 = q_0.docIteratorGetMatch()

            # Other query arguments must match the docid of the
            # first query argument.
            matchFound = True

            for q_i in self._args[ 1: ]:

                q_i.docIteratorAdvanceTo(docid_0)

                # If any argument is exhausted there, are no more matches.
                if not q_i.docIteratorHasMatch(r):	
                    return(False)

                docid_i = q_i.docIteratorGetMatch()

                if docid_0 != docid_i:	# docid_0 can't match.  Try again.
                    q_0.docIteratorAdvanceTo(docid_i)
                    matchFound = False
                    break

            if matchFound:
                self.docIteratorSetMatchCache(docid_0)

        return(True)


    def docIteratorHasMatchFirst(self, r):
        """
        An instantiation of docIteratorHasMatch that is true if the
        query has a document that matches the first query argument;
        some subclasses may choose to use this implementation.

        r: The retrieval model that determines what is a match.
        Returns True if the query matches, otherwise False.
        """

        q_0 = self._args[ 0 ]

        if q_0.docIteratorHasMatch(r):
            docid = q_0.docIteratorGetMatch()
            self.docIteratorSetMatchCache(docid)
            return True
        else:
            return False


    def docIteratorHasMatchMin(self, r):
        """
        An instantiation of docIteratorHasMatch that is true if the
        query has a document that matches at least one query argument;
        the match is the smallest docid to match; some subclasses may
        choose to use this implementation.

        r: The retrieval model that determines what is a match
        Returns True if the query matches, otherwise False.
        """

        minDocid = None

        for q_i in self._args:
            if q_i.docIteratorHasMatch(r):
                if minDocid is None:
                    minDocid = q_i.docIteratorGetMatch()
                else:
                    minDocid = min(minDocid, q_i.docIteratorGetMatch())

        if minDocid is None:
            return False
        else:
            self.docIteratorSetMatchCache(minDocid)
            return True


    def docIteratorHasMatchCache(self):
        """
        Returns True if a match is cached, otherwise False.
        """
        return(self.docIteratorMatchCache != Qry.INVALID_DOCID)


    def docIteratorSetMatchCache(self, docid):
        """
        Set the matching docid cache.

        docid: The internal document id to store in the cache.
        """
        self.docIteratorMatchCache = docid


    def getDisplayName(self):
        """
        Every operator has a display name that can be used by
        toString for debugging or other user feedback.  

        Returns the query operator's display name.
        """
        return(self._displayName)


    def initialize(self, RetrievalModel):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators; this method must be called before iteration
        can begin.

        r: A retrieval model that guides initialization
        throws IOException: Error accessing the Lucene index.
        """
        raise Exception('{}.{} needs to implement {}'.format(
            self.__class__.__name__,
            sys._getframe().f_code.co_name,
            retrievalModel.__class__.__name__))
                         

    def delArg(self, i):
        """
        Delete the i'th argument from the list of query operator arguments.
        """
        del(self._args[ i ])


    def setDisplayName(self, name):
        """
        Every operator must have a display name that can be used by
        toString for debugging or other user feedback.  

        name: The query operator's display name
        """
        self._displayName = name


    def __str__(self):
        """
        Get a string version of this query operator.  This is a generic
        method that works for most query operators.  However, some query
        operators (e.g., #NEAR/n or #WEIGHT) may need to override this
        method with something more specific.

        Returns the string version of this query operator.
        """

        result = ''

        for arg in self._args:
            result += str(arg) + ' '

        return(self._displayName + '(' + result + ')')
