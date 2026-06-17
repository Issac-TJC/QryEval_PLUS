"""
A simple commandline utility for inspecting Lucene 8 indexes.
Run it to see a simple usage message.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


import sys

import PyLu

from Idx import Idx
from InvList import InvList

# ------------------ Global variables ---------------------- #

externalIdField = 'externalId'	# The index field that stores the external id

usage =	(
        "Usage:  python " +
	sys.argv[0] +
	" -index INDEX_PATH\n\n" +
	"where options include\n" +
        "    -list-attribute IDOCID FIELD\n" +
        "\t\t\tdisplay a document attribute" +
        "    -list-doc IDOCID\n" +
	"\t\t\tlist the contents of the document with internal docid\n" +
	"\t\t\tIDOCID\n" +
        "    -list-docids\tlist the external docids of each document\n" +
        "    -list-edocid IDOCID\n" +
        "\t\t\tlist the external docid of the document\n" +
        "\t\t\twith internal docid of IDOCID\n" +
        "    -list-idocid EDOCID\n" +
        "\t\t\tlist the internal docid of the document\n" +
        "\t\t\twith external docid of EDOCID\n" +
	"    -list-fields\tlist the fields in the index\n" +
	"    -list-metadata IDOCID\n" +
	"\t\t\tdisplay the metadata fields for the document\n" +
        "\t\t\twith internal docid of IDOCID\n" +
	"    -list-postings TERM FIELD\n" +
	"\t\t\tdisplay the posting list entries for\n" +
	"\t\t\tterm TERM in field FIELD\n" +
	"    -list-postings-sample TERM FIELD\n" +
	"\t\t\tdisplay the first few posting list entries for\n" +
	"\t\t\tterm TERM in field FIELD\n" +
	"    -list-stats\n" +
	"\t\t\tdisplay corpus statistics\n" +
	"    -list-terms FIELD" +
	"\tdisplay the term dictionary for field FIELD\n" +
	"    -list-termvector IDOCID\n" +
	"\t\t\tdisplay the term vectors for all fields in the document\n" +
	"\t\t\twith internal IDOCID\n" +
	"    -list-termvector-field IDOCID FIELD\n" +
	"\t\t\tdisplay the term vector for FIELD in the document\n" +
	"\t\t\twith internal IDOCID\n")


# ------------------ Methods (sorted alphabetically) ------- #

def checkDocid(docid):
    """
    Print an error mesage if the docid is not valid.

    docid: An internal document id.
    Returns True if the docid is valid, False otherwise.
    """

    if docid < 0 or docid >= Idx.getNumDocs():
        msg_error(f'{docid} is a bad document id.')
        return(False)
    else:
        return(True)


def main():
    """
    """

    # Opening the index first simplifies the processing of the
    # rest of the command line arguments.
    index_is_open = False
    try:
        i = sys.argv.index('-index')
        index_is_open = Idx.open(sys.argv[ i+1 ])
    except:
        msg_error('Error:  Cannot open index')

    if not index_is_open:
        msg_error(usage)
        sys.exit(1)

    # Process the command line arguments. Use a while loop
    # because Python iteration can't consume a variable number
    # of arguments.

    i = 1			# Ignore the script name

    while i < len(sys.argv):

        if sys.argv[ i ] == '-index':
            # -index is handled above, so skip it & its argument.
            i += 1

        elif sys.argv[ i ] == '-list-attribute':
            msg_info('-list-attribute: ')
            if (i+2) >= len(sys.argv):
                msg_error(usage)
                break
            iid = int(sys.argv[ i+1 ])
            field = sys.argv[ i+2 ]
            msg_info(f'Attribute {field} for internal docid {iid}')
            msg_info(Idx.getAttribute(field, iid))

            i += 2

        elif sys.argv[ i ] == '-list-doc':
            msg_info('-list-doc: ')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            docid = int(sys.argv[ i+1 ])
            print(f'\nDocument:  docid {docid}')
            if checkDocid(docid):
                print(Idx.indexReader.document(docid).toString())
            i += 1

        elif sys.argv[ i ] == '-list-docids':
            msg_info('-list-docids:')
            for j in range(0, Idx.getNumDocs()):
                print(f'Internal --> external docid: '
                      f'{j} --> {Idx.getExternalDocid(j)}')

        elif sys.argv[ i ] == '-list-edocid':
            msg_info('-list-edocid: ')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            iid = int(sys.argv[ i+1 ])
            eid = Idx.getExternalDocid(iid)
            msg_info('Internal docid --> External docid: {} --> {}'.format(
                iid, eid))
            i += 1

        elif sys.argv[ i ] == '-list-fields':
            msg_info('-list-fields: ')
            fields = Idx.getFields()
            msg_info(f'\nNumber of fields:  {len(fields)}')
            for field in fields:
                msg_info(f'\t{field}')
            i += 1

        elif sys.argv[ i ] == '-list-idocid':
            msg_info('-list-idocid: ')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            eid = sys.argv[ i+1 ]
            iid = Idx.getInternalDocid(eid)
            msg_info('External docid --> Internal docid: {} --> {}'.format(
                      eid, iid))
            i += 1

        elif sys.argv[ i ] == '-list-metadata':
            msg_info('-list-metadata: ')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            iid = sys.argv[i+1]
            listMetadata (int(iid))
            i += 1

        elif sys.argv[ i ] == '-list-postings':
            msg_info('-list-postings: ')
 
            if (i+2) >= len(sys.argv):
                msg_error(usage)
                break
            term = sys.argv[ i+1 ]
            field = sys.argv[ i+2 ]
            i += 2

            listPostings(term, field)

        elif sys.argv[ i ] == '-list-postings-sample':
            msg_info('-list-postings-sample: ')
 
            if (i+2) >= len(sys.argv):
                msg_error(usage)
                break
            term = sys.argv[ i+1 ]
            field = sys.argv[ i+2 ]
            i += 2

            listPostings(term, field, 5)

        elif sys.argv[ i ] == '-list-stats':
            msg_info('-list-stats: ')
            print('Corpus statistics:')
            print (f'\tnumdocs\t\t{Idx.getNumDocs()}')
            for f in ['url', 'keywords', 'title', 'body', 'inlink']:
                avglen = Idx.getSumOfFieldLengths(f) / Idx.getDocCount (f)
                print (f'\t{f}:\t'
                       f'\tnumdocs= {Idx.getDocCount(f)}'
                       f'\tsumTotalTF={Idx.indexReader.getSumTotalTermFreq(f)}'
                       f'\tavglen={avglen}')

        elif sys.argv[ i ] == '-list-terms':
            msg_info('-list-terms: ')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
 
            listTermDictionary(sys.argv[i+1])
            i += 1

        elif sys.argv[i] == '-list-termvector':
            msg_info('-list-termvector:')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            listTermVectors(sys.argv[i+1])
            i += 1

        elif sys.argv[i] == '-list-termvector-field':
            msg_info('-list-termvector-field:')
            if (i+1) >= len(sys.argv):
                msg_error(usage)
                break
            print(termVectorToStr(Idx.getTermVector(int(sys.argv[i+1]),
                                  sys.argv[i+2])))
            i += 2

        else:
            msg_error('\nWarning:  Unknown argument {} ignored. '.format(
                sys.argv[ i ]))

        i += 1

    # Close the index and exit gracefully.
    Idx.close()


def listMetadata(docid):
    """
    Display the stored fields in a document.

    docid: The internal document id of the document
    """

    print(f'\nMetadata:  docid {docid}')

    if docid < 0 or docid >= Idx.getNumDocs():
        print(f'ERROR:  {docid} is a bad document id.')
        return

    # Iterate over the fields in this document.
    d = Idx.indexReader.document(docid)
    fields = d.getFields()
    fieldIterator = fields.iterator()
    while fieldIterator.hasNext():
        field = fieldIterator.next()
        if field.fieldType().indexOptions() == PyLu.LIndexOptions.DOCS:
            fieldName = field.name()
            fieldValues = d.getValues(fieldName)
            print(f'  Field: {fieldName}'
                  f' length: {len(fieldValues)}')
            for i in range(0, len(fieldValues)):
                print(f'    {fieldValues[i]}')

        
def listPostings (term, field, n=None):
    """
    Displays the first n postings for a term in a field.
    Set n to None to display all postings.

    reader: An IndexReader
    """

    msg_info(f'\nPostings: {term} {field}')

    inv_list = InvList(field, term)

    print(f'\tdf:  {str(inv_list.df)}')
    print(f'\tctf: {str(inv_list.ctf)}')

    if inv_list.df < 1:
        return

    # Iterate through the first n postings.

    if n == None:
        n = Idx.getNumDocs()

    count = 0

    for posting in inv_list.postings:
        count += 1
        if count > n:
            break

        print(f'\tdocid: {posting.docid}')
        print(f'\ttf: {posting.tf}')
        positions = [str(p) for p in posting.positions]
        print(f'\tPositions: {" ".join(positions)}')

    return

def listTermDictionary(fieldName):
    """
    Displays the term dictionary for a field.
    """

    print(f'\nTerm Dictionary:  field {fieldName}')
    print('    ==> Warning: This is very slow in Python <==')

    # Each index segment has its own term dictionary. Merge them. Sigh.
    term_dict = {}
    for leafContext in Idx.indexReader.leaves():
        leafTerms = leafContext.reader().terms(fieldName)
        if leafTerms is None or leafTerms.size () == -1:            
            continue		# No terms in this index segment

        ithTerm = leafTerms.iterator()
        while ithTerm.next() != None:
            term = ithTerm.term().utf8ToString()
            if term not in term_dict:
                term_dict[term] = [0, 0]
                    
            term_dict[term][0] += ithTerm.docFreq()
            term_dict[term][1] += ithTerm.totalTermFreq()

    terms = sorted(term_dict.keys())
    print(f'    Vocabulary size: {len(terms)}')
    for t in terms:
        print(f'      {t:<40} {term_dict[t][0]} {term_dict[t][1]}\n')
        
 
def listTermVectors(docidString):
    """
    Displays the term vectors for all of the fields in a document.
    """

    print(f'\nTermVector:  {docidString} ')

    docid = int(docidString)

    if not checkDocid(docid):
        return

    # Iterate over the fields in this document.
    fields = Idx.getFields()

    for field in fields:
        tv = Idx.getTermVector(docid, field)
        print(termVectorToStr(tv))
        

def msg_error(*args):
  """Print error messages carefully and quickly."""

  text = ' '.join(str(arg) for arg in args)
  print('Error: ' + text)

    
def msg_info(*args):
  """Print informational messages carefully and quickly."""

  text = ' '.join(str(arg) for arg in args)
  print(text)


def msg_warning(*args):
  """Msg warning messages carefully and quickly."""

  text = ' '.join(str(arg) for arg in args)
  print('Warning: ' + text)


def termVectorToStr(tv):
    """Render the term vector as a string. Old InspectIndex format."""

    s = '  Field: {}\n'.format(tv.fieldName)
    s += f'    Stored length: {tv.positionsLength()}\n'

    if tv.positionsLength() > 0:
        s += f'    Vocabulary size: {tv.stemsLength()-1} terms\n'
        s += '      {:>10} {:<19} {:>2} positions\n'.format(' ', 'term', 'tf')

    # Ignore stem 0 (stopwords)
    for stem_i in range(1, tv.stemsLength()):
        s += '      {:>10} {:<19} {:>2} '.format(
            stem_i-1, str(tv.stemString(stem_i)), tv.stemFreq(stem_i))
        for position_j in range(0, tv.positionsLength()):
            if tv.positions[position_j] == stem_i:
                s += '{} '.format(position_j)
        s += '\n'
    
    return s



    
# ------------------ Script body --------------------------- #

if __name__ == '__main__':
    main()
