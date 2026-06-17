"""
Manage query rewriting tasks in the pipeline.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RewriteWithPrf import RewriteWithPrf

class Rewriter:
    """
    Manage query rewriting tasks in the pipeline.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._type = parameters.get('type')
        self._model = None
        
        if self._type and self._type.lower() == 'prf':
            self._model = RewriteWithPrf(parameters)
        else:
            raise AttributeError(f'Unknown Rewriter type: {self._type}')

    def execute(self, batch):
        """
        Execute the query rewriting on the batch.
        
        batch: A dict of {qid: {'qstring': qstring, 'ranking': [...]}}
        Returns the updated batch with rewritten qstrings.
        """
        return self._model.rewrite(batch)