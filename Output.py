"""
Write results in various formats.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json

from TeIn import TeIn
import Util

class Output:

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._type = parameters['type']
        self._outputPath = parameters['outputPath']
        self._outputLength = parameters.get('outputLength')
        self._promptPath = parameters.get('promptPath',
                                          parameters.get('rag:promptPath'))


    def close(self):
        if '_teIn' in vars(self):
            self._teIn.close()


    def execute(self, batch):
        """
        Write the output about the batch to a file

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        if self._type == 'trec_eval':
            teIn = TeIn(self._outputPath, self._outputLength)
            for qid in batch:
                teIn.appendQuery(qid, batch[qid]['ranking'], 'reference')
            teIn.close()
        elif self._type == 'triviaqa_evaluation':
            answers = {}
            prompt_lines = []

            for qid in batch:
                answers[qid] = batch[qid].get('answer', '')

                if self._promptPath is not None and 'prompt_rag' in batch[qid]:
                    prompt_json = json.dumps(batch[qid]['prompt_rag'],
                                             ensure_ascii=True)
                    prompt_lines.append(f'{qid}: {prompt_json}')

            with open(self._outputPath, 'w') as f:
                json.dump(answers, f)

            if self._promptPath is not None:
                Util.file_write_strings(self._promptPath, prompt_lines)
        else:
            raise Exception('Error: Unknown Output format')
            
        return(batch)
