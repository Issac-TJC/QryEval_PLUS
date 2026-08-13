"""
Get an initial ranking for a set of queries. The ranking may come
from an .inRank file or from a bag-of-words ranker (ranked and
unranked boolean, Indri, BM25).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import itertools

from collections import OrderedDict

from qryeval_plus.core import Util

from qryeval_plus.retrieval.RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from qryeval_plus.retrieval.RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from qryeval_plus.retrieval.RetrievalModelBM25 import RetrievalModelBM25

class Ranker:
    """
    Get an initial ranking for a set of queries. The ranking may
    come from an .inRank file or from a bag-of-words ranker (ranked
    and unranked boolean, Indri, BM25).
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model = None
        self._inRank_path = None
        self._is_dense = False
        self._max_results = 1000       		# default

        if 'outputLength' in parameters:
            self._max_results = parameters['outputLength']

        if 'type' not in parameters:
            raise AttributeError('Missing parameter: type')

        if parameters['type'] == 'inRankFile':
            self._inRank_path = parameters['inRankFile:Path']
        elif parameters['type'] == 'UnrankedBoolean':
            self._model = RetrievalModelUnrankedBoolean(parameters)
        elif parameters['type'] == 'RankedBoolean':
            self._model = RetrievalModelRankedBoolean(parameters)
        elif parameters['type'] == 'BM25':
            self._model = RetrievalModelBM25(parameters)
        elif str(parameters['type']).lower() == 'dense':
            from qryeval_plus.retrieval.DenseRanker import DenseRanker

            self._model = DenseRanker(parameters)
            self._is_dense = True
        else:
            raise AttributeError('Unknown type: {parameters["type"]}')


    def execute(self, batch):
        """
        Get rankings for each query. Any prior ranking is ignored.

        batch: A dict of {qid: {'qstring': qstring } ... }

        Return a dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId)] ...}
                          ... }
        """
        if self._model is not None:
            if self._is_dense:
                return(self._model.rerank_batch(batch))
            return(self.get_rankings_bow(batch))
        elif self._inRank_path is not None:
            for qid, ranking in Util.read_rankings(self._inRank_path).items():
                batch[qid]['ranking'] = ranking
            return(batch)
        else:
            raise Exception('Error: Ranker does not know how to rank')
                        

    def get_rankings_bow(self, batch):
        """
        Add  rankings for each query to the batch object. Each ranking is
        a list of (score, externalId) tuples.
        
        batch: A dict of {query_id: {'qstring': query_string}}.
        """
        from qryeval_plus.core.Ranking import Ranking
        from qryeval_plus.query.QryParser import QryParser

        for qid in batch:
            # Prepare to evaluate a query
            qstring = batch[qid]["qstring"]
            print(f'{qid}: {qstring}', flush=True)
            qstring = f'{self._model.defaultQrySop}({qstring})'
            q = QryParser.getQuery(qstring)
            print(f'    ==> {str(q)}', flush=True)
            q.initialize(self._model)
            ranking = Ranking(self._max_results)

            # Evaluate the query. Each pass of the loop finds
            # one matching document.
            while(q.docIteratorHasMatch(self._model)):
                docid = q.docIteratorGetMatch()
                score = q.getScore(self._model)
                q.docIteratorAdvancePast(docid)
                ranking.add(docid, score)

            batch[qid]['ranking'] = ranking.get_ranking()

        return(batch)
