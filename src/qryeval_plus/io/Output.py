"""
Write results in various formats.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json
from pathlib import Path

from qryeval_plus.io.TeIn import TeIn
from qryeval_plus.core import Util

class Output:

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._type = parameters['type']
        self._outputPath = parameters['outputPath']
        self._outputLength = parameters.get('outputLength')
        self._promptPath = parameters.get('promptPath',
                                          parameters.get('rag:promptPath'))
        self._metadataPath = parameters.get('metadataPath')


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
        Path(self._outputPath).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True)
        if self._promptPath is not None:
            Path(self._promptPath).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True)
        if self._metadataPath is not None:
            Path(self._metadataPath).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True)

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

            if self._metadataPath is not None:
                metadata = {
                    qid: {
                        **batch[qid].get('llm', {}),
                        **(
                            {'agent': batch[qid]['agent']}
                            if 'agent' in batch[qid] else {}
                        ),
                    }
                    for qid in batch
                }
                with open(self._metadataPath, 'w') as f:
                    json.dump(metadata, f, indent=2)
        else:
            raise Exception('Error: Unknown Output format')
            
        return(batch)
