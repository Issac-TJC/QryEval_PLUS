# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import cProfile
import json
import os
import sys

from qryeval_plus.core import Util

from qryeval_plus.core.Timer import Timer

# ------------------ Global variables ---------------------- #

usage = "Usage:  python QryEval.py paramFile\n\n";


# ------------------ Methods (alphabetical) ---------------- #

def main ():
    """The main function"""

    timer = Timer()
    timer.start()

    # Initialize the index and experiment parameters.
    parameters = readParameterFile()

    from qryeval_plus.core.Idx import Idx
    from qryeval_plus.io.Output import Output
    from qryeval_plus.rag.Agent import Agent
    from qryeval_plus.rerank.Reranker import Reranker
    from qryeval_plus.retrieval.Ranker import Ranker
    from qryeval_plus.rewrite.Rewriter import Rewriter

    Idx.open(parameters['indexPath'])
    queries = Util.read_queries(parameters['queryFilePath'])

    # Run the ranking pipeline.  In most search engines, the pipeline
    # evaluates 1 query at a time. Some of the tools used in HW2-HW5
    # are faster if called 1 time * n queries instead of n times * 1
    # query, so each stage of our pipeline does all queries and then
    # passes its results to the next stage.
    # 
    # Start by creating an initial batch object. Tasks will update it
    # as they produce new information.
    batch = {qid: {'qstring': qstring} for qid, qstring in queries.items()}
    task_list = [t for t in parameters.keys() if t.startswith('task_')]


    for task_name in task_list:
        print(f'\n-- {task_name}: {parameters[task_name]["type"]} --\n')
        task_type = task_name.split(':')[1]
        task_parameters = parameters[task_name]

        if task_type == 'agent':
            task = Agent(task_parameters)
        elif task_type == 'output':
            task = Output(task_parameters)
        elif task_type == 'ranker':
            task = Ranker(task_parameters)
        elif task_type == 'reranker':
            task = Reranker(task_parameters)
        elif task_type == 'rewriter':
            task = Rewriter(task_parameters)
        else:
            raise Exception(f'Error: Unexpected task type {task_type} in {task_name}')

        batch = task.execute(batch)
    
    # Clean up
    Idx.close()
    timer.stop()
    print('Time:  ' + str(timer))


def readParameterFile():
    """
    Get a dict that contains the contents of the parameter file.
    """

    # Remind the forgetful.
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print('No such file {}.'.format(sys.argv[1]))
        sys.exit(1)

    # Read the .param file into a dict. Values that look like numbers
    # are stored as numbers.
    with open(sys.argv[1]) as f:
        d = json.load(f)
    d = Util.str_to_num(d)

    return(d)


# ------------------ Script body --------------------------- #

if __name__ == '__main__':
    main()
