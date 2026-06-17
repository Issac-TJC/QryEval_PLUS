"""
Create, access, and manipulate document score lists.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


from Idx import Idx

class Ranking:
    """
    A Ranking object stores a search ranking and returns the top n.
    """

    class Entry:
        """
        A utility class to create an <externalDocid, score> object
        for the _ranking list.
        """

        def __init__(self, score, externalId):
            self.externalId = externalId
            self.score = score

        def __lt__(self, other):
            return(self.score < other.score or
                   (self.score == other.score and
                    self.externalId > other.externalId))

        def __str__(self):
            return(f'{self.externalId} {self.score}')


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, n):
        """Create an empty list that can store a ranking."""
        self._max_size = n
        self._ranking = []


    def __len__(self):
        return(len(self._ranking))


    def add(self, internalId, score):
        """
        Add a document to the ranking (unsorted).
        internalId: An internal document id.
        score: The document score.
        """
        externalId = Idx.getExternalDocid(internalId)
        self._ranking.append(self.Entry(score, externalId))
        

    def get_ranking(self):
        """
        Get a ranked list in (score, externalId) order.
        """
        results_qid = [(r.score, r.externalId) for r in self._ranking]
        results_qid.sort(key=lambda r: (-r[0], r[1]))
        return(results_qid[0:self._max_size])


