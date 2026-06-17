import sys
from qryeval_plus.core.InvList import InvList
from qryeval_plus.query.QryIop import QryIop

class QryIopWindow(QryIop):
    """The WINDOW/n operator for all retrieval models."""

    def __init__(self, n):
        """
        Create a new WINDOW/n query node.
        
        Args:
            n: The maximum distance (window size) between query arguments.
        """
        QryIop.__init__(self)
        self.n = n

    def evaluate(self):
        """
        Evaluate the query operator; the result is an internal inverted
        list that may be accessed via the internal iterators.
        """
        self.invertedList = InvList(self._field)

        if len(self._args) == 0:
            return

        # Initialize by loading the first document for all arguments
        for q_i in self._args:
            if not q_i.docIteratorHasMatch(None):
                return # If any arg is empty, Intersection is empty.

        # Loop until one of the argument lists is exhausted
        while True:
            # Identical to QryIopNear
            
            current_docids = [q_i.docIteratorGetMatch() for q_i in self._args]
            max_docid = max(current_docids)
            all_match = True

            for q_i in self._args:
                if q_i.docIteratorGetMatch() < max_docid:
                    q_i.docIteratorAdvanceTo(max_docid)
                    if not q_i.docIteratorHasMatch(None):
                        return 
                
                if q_i.docIteratorGetMatch() != max_docid:
                    all_match = False

            if not all_match:
                continue
            
            # Get positions for all terms in the current doc
            # positions_lists[i] is the list of positions for the i-th argument
            positions_lists = [q_i.docIteratorGetMatchPosting().positions for q_i in self._args]
            
            matches = []
            
            # Indices track the current position we are looking at for each term
            indices = [0] * len(self._args)
            
            while True:
                valid_indices = True
                for i in range(len(self._args)):
                    if indices[i] >= len(positions_lists[i]):
                        valid_indices = False
                        break
                
                if not valid_indices:
                    break 

                current_positions = []
                for i in range(len(self._args)):
                    current_positions.append(positions_lists[i][indices[i]])
                
                min_pos = min(current_positions)
                max_pos = max(current_positions)
                
                # Note: Window/n usually implies (max - min) < n. 
                if (max_pos - min_pos) < self.n:
                    matches.append(max_pos)
                    
                    # Match found
                    for i in range(len(indices)):
                        indices[i] += 1
                else:
                    # No Match: The window is too wide.
                    # Advance ONLY the term with the minimum position to shrink the window.
                    min_idx_in_args = -1
                    min_val = float('inf')
                    
                    for i in range(len(current_positions)):
                        if current_positions[i] < min_val:
                            min_val = current_positions[i]
                            min_idx_in_args = i
                            
                    indices[min_idx_in_args] += 1

            # If found matches, store
            if len(matches) > 0:
                matches.sort()
                self.invertedList.appendPosting(max_docid, matches)

            # Advance All
            for q_i in self._args:
                q_i.docIteratorAdvancePast(max_docid)
                if not q_i.docIteratorHasMatch(None):
                    return