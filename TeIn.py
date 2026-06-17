"""
Write search results in trec_eval format.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from Idx import Idx

class TeIn:
    """
    Write search results in trec_eval format.
    """

    # -------------- Constants and variables --------------- #

    _nonexistent_external_docid = 'Nonexistent_Docid'

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, path, max_results):
        """
        Prepare to write search results for multiple queries to a file in
        trec_eval format.

        path: Where to write the file.
        max_results: Maximum number of results to write for each query.

        """
        self._path = path
        self._handle = open(path, 'wt', newline='')
        self._max_results = int(max_results)

        
    def __write_line(self, qid, docid, rank, score, runid):
        self._handle.write('{} Q0 {} {} {:0.12f} {}\n'.format(
            qid, docid, rank, score, runid)),


    def appendQuery(self, qid, results, runId):
        """Append search results for a query to a trec_eval input file.

        qid: Query identifier.
        results: A list of (score, external_id) tuples.
        runId: Run identifier string.

        throws IOException: Error accessing the Lucene index.
        """

        if results is None or  len(results) < 1:
            self.__write_line(qid,
                               self._nonexistent_external_docid, 1, 0, runId)
        else:
            num_results = min(len(results), self._max_results)
            for i in range(0, num_results):
                score = results[i][0]
                external_id = results[i][1]
                self.__write_line(
                    qid,
                    external_id,
                    i+1,
                    score,
                    runId)
            self._handle.flush()


    def close(self):
        self._handle.close()
        self._handle = None
