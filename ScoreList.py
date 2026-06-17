"""
Create, access, and manipulate document score lists.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


from Idx import Idx

class ScoreList:
    """
    Create, access, and manipulate document score lists.
    """

    # ----------------- Nested classes --------------------- #

    class ScoreListEntry:
        """
        A utility class to create a <internalDocid, score> object.
        """

        def __init__(self, internalDocid, score):
            self.docid = internalDocid
            self.score = score


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        self.scores = []		# A list of ScoreListEntry


    def __len__(self):
        return(len(self.scores))


    def add(self, docid, score):
        """
        Append a document score to a score list.
        docid: An internal document id.
        score: The document's score.
        """
        self.scores.append(self.ScoreListEntry(docid, score))


    def getDocid(self, n):
        """Get the internal docid of the n'th entry."""
        return(self.scores[n].docid)


    def getDocidScore(self, n):
        """Get the score of the n'th entry."""
        return(self.scores[n].score)


    def cmp_to_key (mycmp):
        '''Convert a cmp= function into a key= function'''
        class K:
            def __init__(self, obj, *args):
                self.obj = obj
            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0
            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0
            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0
            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0
            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0
            def __ne__(self, other):
                return mycmp(self.obj, other.obj) != 0
        return K
