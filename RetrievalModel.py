"""
RetrievalModel is the root class in the retrieval model hierarchy. 
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

class RetrievalModel:
    """
    RetrievalModel is the root class in the retrieval model hierarchy. 
    This hierarchy is used to create objects that provide fast access to
    retrieval model parameters and indicate to the query operators how
    the query should be evaluated.
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __str__(self):
        """Human readable information about the object."""
        return(str(self.__dict__))


    def defaultQrySopName():
        """
        The name of the default query operator for the retrieval model.

        Returns the name of the default query operator.
        """
        raise Exception('Each RetrievalModel subclass  needs to implement ' +
                        sys._getframe().f_code.co_name)
