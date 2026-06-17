"""
Manage and provide access to a Lucene index through Java. The Java
interface can be either jpype or jnius. Both are supported. jpype is
slower.

The decision about which interface to use is hard-wired into the
source code (see the java_interface variable).

The java_classpath is also hard-wired (see the java_classpath variable).

"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

# -------------- Configure the Java interface ---------------#

java_interface = 'jnius'		# jpype or jnius
java_classpath = ['.', 'LIB_DIR/*']

# -------------- Implement the Java interface ---------------#

if java_interface == 'jpype':
    import jpype
    from jpype import *
    if not jpype.isJVMStarted():
        jpype.startJVM('-Xmx6g', classpath = java_classpath)

elif java_interface == 'jnius':
    if 'jnius_config' not in sys.modules:	# Loading starts the JVM
        import jnius_config
        jnius_config.add_options('-Xmx6g')
        jnius_config.set_classpath(*java_classpath)
        import jnius
        from jnius import autoclass


def get_jclass(class_hierarchy_path):
    """Get an interface to a Java class."""
    if java_interface == 'jpype':
        return(jpype.JClass(class_hierarchy_path))
    else:
        return(autoclass(class_hierarchy_path))


def nop(arg):
    """No operation. Just return the argument."""
    return(arg)


# ------------------ Java datatypes ------------------------ #

JBoolean = get_jclass('java/lang/Boolean')
JPaths = get_jclass('java.nio.file.Paths')
JStringReader = get_jclass('java.io.StringReader')
JTrue = JBoolean('true')

if java_interface == 'jpype':
    JString = nop
else:
    JString = get_jclass('java.lang.String')

    
# ------------------ Lucene classes ------------------------ #

LBytesRef = get_jclass('org.apache.lucene.util.BytesRef')
LCharTermAttribute = get_jclass('org.apache.lucene.analysis.tokenattributes.CharTermAttribute')
LDirectoryReader = get_jclass('org.apache.lucene.index.DirectoryReader')
LDocIdSetIterator = get_jclass('org.apache.lucene.search.DocIdSetIterator')
LEnglishAnalyzerConfigurable = get_jclass('org.apache.lucene.analysis.en.EnglishAnalyzerConfigurable')
LEnglishAnalyzerConfigurableStemmerType = get_jclass('org.apache.lucene.analysis.en.EnglishAnalyzerConfigurable$StemmerType')
LFieldInfos = get_jclass('org.apache.lucene.index.FieldInfos')
LFSDirectory = get_jclass('org.apache.lucene.store.FSDirectory')
LIndexOptions = get_jclass('org/apache/lucene/index/IndexOptions')
LLeafReaderContext = get_jclass('org.apache.lucene.index.LeafReaderContext')
LPostingsEnum = get_jclass('org.apache.lucene.index.PostingsEnum')
LTerm = get_jclass('org.apache.lucene.index.Term')
LTerms = get_jclass('org.apache.lucene.index.Terms')
LTermsEnum = get_jclass('org.apache.lucene.index.PostingsEnum')
LTokenStream = get_jclass('org.apache.lucene.analysis.TokenStream')


# ------------------ RankLib classes ----------------------- #

RankLib = get_jclass('ciir.umass.edu.eval.Evaluator')


# ------------------ Java QryEval classes ------------------ #

QjIdx = get_jclass('Idx')
QjTermVector = get_jclass('TermVector')
