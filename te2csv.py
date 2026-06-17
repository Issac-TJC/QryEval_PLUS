# Use one or more .teOut files to create a .csv file. All .teOut files
# must be the same length. The script assumes (does not check) that all
# .teOut files are comparable (e.g., same queries, same metrics).
#
# The output file is te2csv.csv.
#
# Usage: python te2csv.py <path_prefix>
#
#        python te2csv.py ./
#        python te2csv.py OUTPUT_DIR/
#        python te2csv.py OUTPUT_DIR/HW1-Train
#        python te2csv.py OUTPUT_DIR/HW1-Train-10
#
# Copyright (c) 2025, Carnegie Mellon University.  All Rights Reserved.

import os
import sys

# ------------------ Global variables ---------------------- #

file_out = 'te2csv.csv'

# ------------------ Methods (sorted alphabetically) ------- #

def get_teOut(filename):
    try:
        with open (filename, 'r') as f:
            lines = f.read().splitlines()
        return([ line.split() for line in lines ])
    except Exception as e:
        print(f'{str (e)}')
        return(None)
    

# ------------------ Script body --------------------------- #

# Remind the forgetful
if len(sys.argv) < 2:
    raise Exception(f'Usage: {sys.argv[0]} path-prefix\n')

# Read the .teOut files
dir_in, prefix = os.path.split(sys.argv[1])
filenames = [ f for f in sorted(os.listdir(dir_in))
              if f.startswith(prefix) and f.endswith('.teOut') ]
teOuts = [ get_teOut(os.path.join(dir_in, file)) for file in filenames ]

# Minimal error checking
if len(set([ len(teOut) for teOut in teOuts ])) != 1:
    raise Exception('.teOut files must be the same length')

# Assemble the .csv
metrics = ['metric'] + [ teOut[0] for teOut in teOuts[0] ]
qids =    ['qid'] +    [ teOut[1] for teOut in teOuts[0] ]
values = [ [filenames[i]] + [ line[2] for line in teOuts[i] ]
           for i in range(len(teOuts)) ]
rows = zip(metrics, qids, *values)
lines = [ ','.join(row) for row in rows ]
with open (file_out, 'w') as file_out:
    for line in lines:
        file_out.write (line + '\n')
