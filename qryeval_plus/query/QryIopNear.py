import sys
from qryeval_plus.core.InvList import InvList
from qryeval_plus.query.QryIop import QryIop

class QryIopNear(QryIop):
    """The NEAR/n operator for all retrieval models."""
    """Fail in map metric, because not all the iterators are advanced when a match is found."""
    """Follow https://github.com/MelonTennis/SearchEngine/blob/master/QryIopNear.java."""

    def __init__(self, n):
        """
        Create a new NEAR/n query node.
        
        Args:
            n: The maximum distance between query arguments.
        """
        QryIop.__init__(self)
        self.n = n

    def evaluate(self):
        """
        Evaluate the query operator; the result is an internal inverted
        list that may be accessed via the internal iterators.
        """
        # Create an empty inverted list.
        self.invertedList = InvList(self._field)

        if len(self._args) == 0:
            return

        while self.docIteratorHasMatchAll(None):
            docid = self._args[0].docIteratorGetMatch()
            positions = self.__withinDoc()
            if len(positions) > 0:
                self.invertedList.appendPosting(docid, positions)

            # Advance the first iterator past the current document to continue the loop.
            # This triggers the re-evaluation of docIteratorHasMatchAll in the next iteration.
            self._args[0].docIteratorAdvancePast(docid)

    def __withinDoc(self):
        """
        Finds all match positions within the current document.
        Equivalent to the Java 'withinDoc' method.
        """
        matches = []
        
        positions_lists = [q.docIteratorGetMatchPosting().positions for q in self._args]
        
        # 'indices' acts as the cursor/iterator for each term's position list.
        # indices[i] points to the current position index for term i.
        indices = [0] * len(self._args)

        while not self.__outOfDoc(indices, positions_lists):
            match_pos = self.__findMatch(indices, positions_lists)
            
            if match_pos != -1:
                matches.append(match_pos)
                
                # Advance all iterators.
                # Java: for (Qry arg : this.args) { ((QryIop) arg).locIteratorAdvance(); }
                # This ensures we find non-overlapping sequences.
                for i in range(len(indices)):
                    indices[i] += 1
        
        return matches

    def __outOfDoc(self, indices, positions_lists):
        """
        Checks if any term has run out of positions.
        Equivalent to the Java 'outofDoc' method.
        """
        for i in range(len(indices)):
            if indices[i] >= len(positions_lists[i]):
                return True
        return False

    def __findMatch(self, indices, positions_lists):
        """
        Attempts to find a valid NEAR sequence starting from current indices.
        Equivalent to the Java 'findMatch' method.
        
        Returns:
            The position of the last term if a match is found, otherwise -1.
            (Side effect: updates 'indices' array to advance cursors during search)
        """
        # Iterate from the second term (index 1) to the last term
        for i in range(1, len(self._args)):
            curr_term_pos = positions_lists[i][indices[i]]
            prev_term_pos = positions_lists[i-1][indices[i-1]]

            if curr_term_pos <= prev_term_pos:
                indices[i] += 1
                return -1

            if (curr_term_pos - prev_term_pos) > self.n:
                indices[i-1] += 1
                return -1

        last_idx = len(self._args) - 1
        return positions_lists[last_idx][indices[last_idx]]