"""Manage and provide access to a Lucene index."""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import gzip
import os
import sys

import PyLu


class Idx:
    """
    Manage and provide access to a standard Lucene index or a
    Lucene index that has been augumented with QryEval cache files to
    improve the speed of Python software.  Access to Lucene's Java
    libraries is managed by the PyLu module.
    """


    # -------------- Constants and static variables -------- #

    # Field lengths are expensive to get from the Lucene index.
    # The field length cache (flc) stores field lengths for the
    # current document in case several terms appear in the same
    # field (as is common).
    _flc_doc = None
    _flc_lengths = {}

    # Field lengths and external document ids are expensive to
    # get from the Lucene index. The Lucene data caches (ldc)
    # read this information from a file and store it in Python
    # space for fast access.
    _ldc_eid = None
    _ldc_field_lengths = None
    _ldc_filename_doclengths = 'Idx.pycache.flength.gz'
    _ldc_filename_eids = 'Idx.pycache.eid.gz'

    _externalIdField = 'externalId'
    _JexternalIdField = PyLu.JString( _externalIdField )

    indexReader = None


    # --------------- Internal classes --------------------- #

    class LeafContextCache:
        """
        IndexReader LeafContexts are cached to reduce index calls to
        jnius. Some retrieval models access LeafContexts often when
        looking up basic statistics, which is computationally expensive.
        The cache stores the LeafContexts and commonly accessed attributes
        and values.
        """
        cache = []

        @staticmethod
        def open( indexReader ):
            """
            Create (or reload) the LeafContext cache. This should be
            done whenever an index is opened.
            """
            Idx.LeafContextCache.cache = []

            fields = Idx.getFields()

            for leafContext in indexReader.leaves():
                lcc = {}
                lcc[ 'leaf_context' ] = leafContext
                lcc[ 'min_docid' ] = leafContext.docBase
                lcc[ 'num_docs' ] = leafContext.reader().numDocs()
                lcc[ 'leaf_reader' ] = leafContext.reader()
                Idx.LeafContextCache.cache.append( lcc )


        @staticmethod
        def getByIdocid( docid ):
            """Get cached information about a LeafContext."""
            leafContext = None
            for lcc in Idx.LeafContextCache.cache:
                if ( docid >= lcc[ 'min_docid' ] and
                     docid <  lcc[ 'min_docid' ] + lcc[ 'num_docs' ] ):
                    return( lcc )

            raise Exception( 'No cached leaf context for docid {}.'.format(
                docid ) )


        @staticmethod
        def getByEdocid( docid ):
            """Get cached information about an external docid."""

            term = PyLu.LTerm ( Idx._JexternalIdField, PyLu.JString( docid ) )

            if Idx.indexReader.docFreq( term ) > 1:
                raise Exception( 'Multiple matches for external id ' + docid )

            for lcc in Idx.LeafContextCache.cache:
                if lcc[ 'leaf_reader' ].postings( term ) != None:
                    return( lcc )

            raise Exception( 'No cached leaf context for external id {}.'.format(
                docid ) )


    # -------------- Methods (alphabetical) ---------------- #

    @staticmethod
    def __get_cache_eids( index_path ):
        """Read and return a cache of document external ids."""

        try:
            path = os.path.join( index_path, Idx._ldc_filename_eids )
            f = gzip.open( path, 'rb' )
            contents = f.read()
            f.close()
        except Exception as e:
            print( 'Cannot open file', Idx._ldc_filename_eids )
            print( str (e) )
            return( None )

        contents = [ c.decode() for c in contents.split( '\n'.encode() ) ]
        Idx._ldc_eid = contents[ 1 : ]


    @staticmethod
    def __get_cache_fieldlengths( index_path ):
        """Read and return a cache of document field lengths."""

        try:
            path = os.path.join( index_path, Idx._ldc_filename_doclengths )
            f = gzip.open( path, 'rb' )
            contents = f.read()
            f.close()
        except Exception as e:
            print( 'Cannot open file', Idx._ldc_filename_doclengths )
            print( str (e) )
            return( None )

        # Initial processing of the file contents
        contents = [ c.decode() for c in contents.split( '\n'.encode() ) ]

        field_names = contents[ 0 ].split( ',' )
        corpus_size = int( contents[ 1 ] )

        # Create an empty cache
        cached_lengths = {}
        for f in field_names:
            cached_lengths[ f ] = []

        # Populate the field length cache
        for d_i in range( 2, corpus_size + 2 ):
            field_lengths = contents[ d_i ].split( ',' )
            for f_j in range( 0, len( field_names ) ):
                key = field_names[ f_j ]
                value = int( field_lengths[ f_j ] )
                cached_lengths[ key ].append( value )

        Idx._ldc_field_lengths = cached_lengths


    @staticmethod
    def close():
        """
        Close the open index.
        """
        Idx.indexReader.close()


    @staticmethod
    def getAttribute( attributeName, docid ):
        """
        Get an attribute for a document, or None.

        attributeName: Name of a document attribute.
        docid: An internal document id (an integer).
        """
        return(PyLu.QjIdx.getAttribute(PyLu.JString(attributeName), docid))



    @staticmethod
    def getDocCount( fieldName ):
        """
        Get the number of documents that contain a specified field.

        fieldName: The name of a document field.
        """
        return( Idx.indexReader.getDocCount( PyLu.JString( fieldName ) ) )
  
  
    @staticmethod
    def getDocFreq( fieldName, term ):
        """
        Get the document frequency (df) of a term in a field (e.g.,
        the number of documents that contain 'apple' in title fields).

        fieldName: The name of a document field.
        term: A lexically-processed term that may appear in the corpus.
        """
        b = PyLu.LBytesRef( PyLu.JString( term ) )
        t = PyLu.LTerm( PyLu.JString( fieldName ), b )
        return( Idx.indexReader.docFreq( t ) )


    @staticmethod
    def getExternalDocid( iid ):
        """
        Get the external document id for a document specified by an
        internal document id.

        iid: An internal document id (an integer).
        """
        if Idx._ldc_eid is not None:
            return( Idx._ldc_eid[ iid ] )

        d = Idx.indexReader.document( iid )
        return( str( d.get( Idx._JexternalIdField ) ) )


    @staticmethod
    def getFields():
        """
        Get a list of document fields supported by this index.
        """
        fields = []
    
        for f in PyLu.LFieldInfos.getMergedFieldInfos( Idx.indexReader ):
            fields.append( f.name )

        return( fields )


    @staticmethod
    def getFieldLength( fieldName, docid ):
        """
        Get the length of a field in a document. The length includes stopwords.

        fieldName: The name of a document field.
        docid: An internal document id (an integer).
        """

        # The Lucene data cache is fastest, so check it first.
        if Idx._ldc_field_lengths is not None:
            return( Idx._ldc_field_lengths[ fieldName ][ docid ] )

        # Check the cache for the field length.
        if docid != Idx._flc_doc:		# Is the cache outdated?
            Idx._flc_lengths = {}
            Idx._flc_doc = docid

        if fieldName in Idx._flc_lengths:	# Is the field in the cache?
            return( Idx._flc_lengths[ fieldName ] )
            
        # Get the field length from the Lucene index.
        field_length = 0

        lc_cache = Idx.LeafContextCache.getByIdocid( docid )
        leafReader = lc_cache[ 'leaf_reader' ]
        leafDocid = docid - lc_cache[ 'min_docid' ]
        norms = lc_cache[ 'leaf_reader' ].getNormValues( fieldName )

        if norms != None:
            if norms.advanceExact (leafDocid):
                field_length = norms.longValue()

        # Cache the field length in case it is needed again.
        Idx._flc_lengths[ fieldName ] = field_length

        return( field_length )


    @staticmethod
    def getInternalDocid( docid ):
        """
        Get the internal document id for a document specified by its
        external id, e.g. clueweb09-enwp00-88-09710.

        docid: An external document id (a string).
        """
        return(PyLu.QjIdx.getInternalDocid(PyLu.JString(docid)))


    @staticmethod
    def getNumDocs():
        """
        Get the total number of documents in the corpus.
        """
        return Idx.indexReader.numDocs()


    @staticmethod
    def getSumOfFieldLengths( fieldName ):
        """
        Get the total number of term occurrences contained in all
        instances of the specified field in the corpus (e.g., add up
        the lengths of every TITLE field in the corpus).

        fieldName: The name of a document field.

        Returns the total number of term occurrences.

        """
        return( Idx.indexReader.getSumTotalTermFreq( PyLu.JString( fieldName ) ) )


    @staticmethod
    def getTermVector(docid, fieldName):
        """
        Return an Indri DocVector-style interface to the Lucene
        termvector for a field in a document. The TermVector has
        an internal "stems" vector (a list of the terms that occur
        in this document field) and an internal "positions" vector
        (the "stems" index of the term that occurs at each position).

        docid: An internal document id.
        fieldName: The name of a document field.


        A (Java) TermVector object is returned. The TermVector
        has the following fields and methods.

        Fields:
            docid	The internal id of the document
            fieldName	The name of the document field

        Methods:
            indexOfStem(stem)
                        Get the index of stem in the stems vector,
                        or -1 if the stems vector does not contain the stem.
            positionsLength()
			Get the number of positions in this field
			(the length of the field). If positions are not
                        stored, return 0.
                        Note: Idx.getFieldLength report a longer length
                        if the field ends with stopwords.
            stemAt(i)
			Return the index of the stem that occurred at
			position i in the document.
	    stemDf(i)
			Returns the df of the i'th stem.
            stemFreq(i)
			Get the frequency of the n'th stem in the current
			doc, or -1 if the index is invalid. The frequency
                        for stopwords (i=0) is not stored (0 is returned).
            stemsLength()
			The number of unique stems in this field.
	    stemString(i)
			Get the string for the i'th stem, or null if the
			index is invalid.
            totalStemFreq(i)
			Returns ctf of the i'th stem.
        """
        return(PyLu.QjTermVector(docid, PyLu.JString(fieldName)))


    @staticmethod
    def getTotalTermFreq(fieldName, term):
        """
        Get the collection term frequency (ctf) of a term in
        a field (e.g., the total number of times the term 'apple'
        occurs in title fields.

        fieldName: The name of a document field.
        term: A lexically-processed term that may appear in the corpus.

        Returns the total number of term occurrence.
        """
        return(PyLu.QjIdx.getTotalTermFreq(PyLu.JString(fieldName),
                                           PyLu.JString(term)))


    @staticmethod
    def open (index_path, Idxpycache=True):
        """
        Open a Lucene index.

        indexPath: A path to a directory that contains a Lucene index.
        Idxpycache: Iff True, Idx.pycache.xxx files are used, if available.

        Returns True if the index was opened, otherwise returns False.
        """

        try:
            # Lucene needs an absolute path
            if not os.path.isabs(index_path):
                index_path = os.path.abspath( index_path )

            # Open the index for access by Python code
            p = PyLu.JPaths.get ( index_path )
            fsd = PyLu.LFSDirectory.open(p)
            dr = PyLu.LDirectoryReader.open(fsd)
            Idx.indexReader = dr
            Idx.LeafContextCache.open( dr )

            if Idxpycache:
                Idx.__get_cache_eids( index_path )
                Idx.__get_cache_fieldlengths( index_path )

            # Open the index for access by Java code
            PyLu.QjIdx.open(index_path)

            return( True )
        except Exception as e:
            print(f'Error: {str(e)}')
            return( False )

