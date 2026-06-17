"""
Define and store data for the Ranked Boolean Retrieval Model.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RetrievalModel import RetrievalModel

class RetrievalModelRankedBoolean(RetrievalModel):
    """
    Define and store data for the Ranked Boolean Retrieval Model.
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        RetrievalModel.__init__(self)		# Inherit from RetrievalModel
        # self.defaultQrySop = '#OR'
        self.defaultQrySop = '#AND'
