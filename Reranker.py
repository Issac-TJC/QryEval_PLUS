"""
Rerank initial rankings for a set of queries. The rankings may
come from an .inRank file or from a bag-of-words ranker (ranked
and unranked boolean, Indri, QL, BM25).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RerankWithLtr import RerankWithLtr
from RerankWithBERT import RerankWithBERT

class Reranker:
    """
    Rerank initial rankings for a set of queries. The rankings may
    come from an .inRank file or from a bag-of-words ranker (ranked
    and unranked boolean, Indri, BM25).
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model = None
        self._rerank_depth = parameters.get('rerankDepth', 1000)

        if 'type' not in parameters:
            raise Exception('Error: Missing parameter type.')
        
        models = {
            'ltr': RerankWithLtr,
            'bertrr': RerankWithBERT
        }
        if parameters['type'].lower() not in models:
            raise Exception('Error: Unknown type: {parameters["type"]}')
        self._model = models[parameters['type'].lower()](parameters)


    def execute(self, batch):
        """
        Rerank the rankings for each query. 

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId)] ...}
                          ... }
        """
        top_batch = {qid: {
            'qstring': batch[qid]['qstring'],
            'ranking': batch[qid]['ranking'][:self._rerank_depth]}
            for qid in batch}
        top_batch = self._model.rerank(top_batch)

        # Merge the top of the reranking with the bottom of the
        # original ranking.
        for qid in batch:
            old_ranking = batch[qid]['ranking']
            new_ranking = top_batch[qid]['ranking']

            if len(old_ranking) > len(new_ranking):

                # If unchanged scores are >= to reranked scores,
                # reduce them so that reranked scores are higher.
                last_reranked = new_ranking[-1][0]
                first_unchanged = old_ranking[self._rerank_depth][0]
                score_adjust = max(0.0,
                                   (first_unchanged + 0.1) - last_reranked)

                # Merge the bottom of old_ranking into new_ranking.
                for i in range(self._rerank_depth, len(old_ranking)):
                    new_score = old_ranking[i][0] - score_adjust
                    docid = old_ranking[i][1]
                    new_ranking.append((new_score, docid))
                
            batch[qid]['ranking'] = new_ranking

        return(batch)
