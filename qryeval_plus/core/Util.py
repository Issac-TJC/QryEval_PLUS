"""
Handy utilities.
"""

import re 

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


def file_read_strings(path):
    """
    Read a file into a list of strings. If the file cannot be
    read, print an error message and return an empty list.
    """
    try:
        f = open (path, 'r')
        contents = f.read().splitlines()
        f.close ()
        return contents
    except Exception as e:
        print(f'Error: Cannot read {path}\n    {str(e)}')
        return None


def file_write_strings(path, lst):
    """
    Write a list of strings (or things that can be converted to
    strings) to a file. If the file cannot be written, print an
    error message.

    path: A file path
    lst: A list of strings or things that can be converted to strings.
    """
    try:
        f = open (path, 'w')
        for l in lst:
            f.write(str(l) + '\n')
    except Exception as e:
        print(f'Error: Cannot write {path}\n    {str(e)}')
        return None


def lower_keys(obj):
    """
    Convert keys in a dict (and nested dicts) to lowercase.
    """
    if type(obj) is not dict:
        return(obj)

    # Iterate over a list of keys, instead of iterating over dict keys
    # directly, because obj may change in the loop.
    for k in list(obj.keys()):
        v = obj[k]
        if type(k) is str:
            if not k.islower():
                del obj[k]
                k = k.lower()
                obj[k] = v
        if type(v) is dict:
            obj[k] = lower_keys(obj[k])

    return(obj)
        
            
def read_queries(path):
    """
    Read a file in .qry format. Return it as a dict of qid:query.
    """
    queries = file_read_strings(path)
    queries = [q.split(":") for q in queries]
    return({q[0].strip(): q[1].strip() for q in queries})    

def read_qrels(path):
    """
    Read a file in .qrels format. Return it as a list [qid, eid, rel].
    """
    qrels = file_read_strings(path)
    return([q.split(maxsplit=3) for q in qrels])


def read_rankings(path, max_rank=None):
    """
    Get a list of rankings for a set of queries. Each ranking is
    a list of (score, externalId) tuples.

    queries: A dict of {query_id:query_string}.
    max_rank: Optional. If provided, ignore all documents beyond max_rank.
    """
    if path is None:
        return({})
    
    last_qid = None
    last_ranking = []
    results = {}

    for line in file_read_strings(path):
        qid, _, eid, rank, score, _ = line.split(maxsplit=5)
        if last_qid != qid:
            if last_qid is not None:
                results[last_qid] = last_ranking
            last_qid, last_ranking = qid, []
        if (max_rank == None) or (int(rank) <= max_rank):
            last_ranking.append((float(score), eid))
    if last_qid is not None:
        results[last_qid] = last_ranking
    return(results)


def str_to_num(obj):
    """
    Convert strings that look like numbers to int or float. If obj is
    a list or a dict, call recursively on the list elements or dict
    values. Objects that cannot be converted are returned unchanged.
    """
    if type(obj) is int or type(obj) is float:
        return(obj)
    elif type(obj) is str:
        try:
            return(int(obj))
        except ValueError:
            try:
                return(float(obj))
            except ValueError:
                return(obj)
    elif type(obj) is dict:
        for k,v in obj.items():
            obj[k] = str_to_num(v)
        return(obj)
    elif type(obj) is list:
        return([str_to_num(x) for x in obj])
    else:
        raise Exception(f'Not yet implemented for {str(type(obj))}')
            
