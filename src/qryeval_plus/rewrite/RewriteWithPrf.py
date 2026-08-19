"""
Implements Pseudo Relevance Feedback (PRF) for Query Rewriting.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
from collections import Counter
from functools import cache
from qryeval_plus.core.Idx import Idx
from qryeval_plus.query.QryParser import QryParser

class RewriteWithPrf:
    """
    Pseudo Relevance Feedback using Okapi or RM3 (Query Likelihood).
    """
    @staticmethod
    @cache
    def _cached_doc_freq(field, term):
        return Idx.getDocFreq(field, term)

    @staticmethod
    @cache
    def _cached_total_term_freq(field, term):
        return Idx.getTotalTermFreq(field, term)

    @staticmethod
    @cache
    def _cached_doc_count():
        return Idx.getNumDocs()

    @staticmethod
    @cache
    def _cached_sum_field_lengths(field):
        return Idx.getSumOfFieldLengths(field)

    @staticmethod
    @cache
    def _cached_okapi_rsj(field, term):
        """Cache the entire RSJ calculation since N and df_t are constant per term/field."""
        N = RewriteWithPrf._cached_doc_count()
        df_t = RewriteWithPrf._cached_doc_freq(field, term)
        
        numerator = N - df_t + 0.5
        denominator = df_t + 0.5
        
        if denominator <= 0:
            return 0.0
            
        return max(0.0, math.log(numerator / denominator))

    @staticmethod
    @cache
    def _cached_rm3_idf_factor(field, term):
        """Cache the IDF factor calculation since ctf and collection_len are constant per term/field."""
        ctf = RewriteWithPrf._cached_total_term_freq(field, term)
        collection_len = RewriteWithPrf._cached_sum_field_lengths(field)
        
        if ctf <= 0 or collection_len <= 0:
            return 0.0
            
        p_t_c = ctf / float(collection_len)
        return math.log(1.0 / p_t_c)

    @staticmethod
    @cache
    def _cached_document_terms(docid, field):
        """Read indexed term vectors or reconstruct them from the stored field."""
        tv = Idx.getTermVector(docid, field)
        if tv is not None and tv.stemsLength() > 0 and tv.positionsLength() > 0:
            frequencies = {
                str(tv.stemString(stem_i)): int(tv.stemFreq(stem_i))
                for stem_i in range(1, tv.stemsLength())
                if tv.stemString(stem_i) is not None
            }
            return frequencies, int(tv.positionsLength()), "term_vector"

        stored = Idx.getAttribute(field + "-string", docid) or ""
        terms = [term for term in QryParser.tokenizeString(stored) if term]
        return dict(Counter(terms)), len(terms), "stored_field"


    def __init__(self, parameters):
        # Default values as specified in the Design Guide
        self.algorithm = parameters.get('prf:algorithm', 'rm3').lower()
        self.num_docs = int(parameters.get('prf:numDocs', 10))
        self.num_terms = int(parameters.get('prf:numTerms', 10))
        self.exp_field_in = parameters.get('prf:expansionFieldIn', 'body')
        self.exp_field_out = parameters.get('prf:expansionFieldOut', 'body')
        self.min_term_length = int(parameters.get('prf:minTermLength', 3))
        self.max_doc_freq_ratio = float(parameters.get('prf:maxDocFreqRatio', 0.25))

        # Original weight: if missing, treat as 0
        self.orig_weight = float(parameters.get('prf:rm3:origWeight', 0.0))
        self.rm3_mu = float(parameters.get('prf:rm3:mu', 1000.0))
        if self.rm3_mu < 0:
            raise ValueError("prf:rm3:mu must be non-negative")
        
        self.out_file = parameters.get('prf:expansionQueryFile')
        if self.out_file:
            self.qry_out_handle = open(self.out_file, 'w')
        else:
            self.qry_out_handle = None

    def __del__(self):
        if self.qry_out_handle:
            self.qry_out_handle.close()


    def _is_valid_term(self, term):
        """
        Discard terms that contain a period, comma, or non-ASCII characters.
        """
        if '.' in term or ',' in term:
            return False
        if not term.isascii():
            return False
        if not term.isalpha() or len(term) < self.min_term_length:
            return False
        doc_count = RewriteWithPrf._cached_doc_count()
        if doc_count and (
            RewriteWithPrf._cached_doc_freq(self.exp_field_in, term) / doc_count
            > self.max_doc_freq_ratio
        ):
            return False
        return True


    def rewrite(self, batch):
        """
        Update the query strings in the batch using PRF.
        """
        for qid in batch:
            original_qstring = batch[qid]['qstring']
            initial_ranking = batch[qid].get('ranking', [])
            
            top_docs = initial_ranking[:self.num_docs]
            internal_docids = [Idx.getInternalDocid(eid) for score, eid in top_docs]
            doc_scores = [score for score, eid in top_docs]
            
            term_scores = self._score_terms(internal_docids, doc_scores)
            
            # Top M Selection: sort descending by score, ascending by term string
            sorted_terms = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
            top_m_terms = sorted_terms[:self.num_terms]
            
            # Format
            top_m_terms.reverse()
            
            # body？url？title？keywords？inlink？ --- IGNORE ---
            suffix_out = f".{self.exp_field_out}" if self.exp_field_out != 'body' else ""
            
            if self.algorithm == 'okapi':
                learned_parts_out = []
                for term, score in top_m_terms:
                    learned_parts_out.append(f"{term}{suffix_out}")
                learned_query_out_str = "#SUM( " + " ".join(learned_parts_out) + " )"
            else: # RM3
                learned_parts_out = []
                for term, score in top_m_terms:
                    learned_parts_out.append(f"{score} {term}{suffix_out}")
                learned_query_out_str = "#WSUM( " + " ".join(learned_parts_out) + " )"
            
            if self.qry_out_handle:
                self.qry_out_handle.write(f"{qid}: {learned_query_out_str}\n")
                self.qry_out_handle.flush()
            
            if self.algorithm == 'okapi':
                learned_parts_internal = []
                for term, score in top_m_terms:
                    learned_parts_internal.append(f"{term}.{self.exp_field_out}")
                learned_query_internal = "#SUM( " + " ".join(learned_parts_internal) + " )"
            else: # RM3
                learned_parts_internal = []
                for term, score in top_m_terms:
                    learned_parts_internal.append(f"{score} {term}.{self.exp_field_out}")
                learned_query_internal = "#WSUM( " + " ".join(learned_parts_internal) + " )"
            
            # 5. Combine with original query for the internal Task 3 Ranker
            if self.orig_weight > 0.0:
                new_weight = 1.0 - self.orig_weight
                # Natural-language questions may contain unmatched parentheses or
                # other query-language punctuation.  Tokenize them before nesting
                # them inside the structured PRF query so user text can never
                # change the operator tree.
                original_terms = QryParser.tokenizeString(original_qstring)
                if not original_terms:
                    raise RuntimeError(
                        "PRF cannot combine feedback with an empty original query"
                    )
                original_query_internal = "#SUM( " + " ".join(
                    f"{term}.body" for term in original_terms
                ) + " )"
                expanded_qstring = (
                    f"#WSUM( {self.orig_weight} {original_query_internal} "
                    f"{new_weight} {learned_query_internal} )"
                )
            else:
                expanded_qstring = learned_query_internal
                
            # 6. Update the batch
            batch[qid]['original_qstring'] = original_qstring
            batch[qid]['qstring'] = expanded_qstring
            
        return batch


    def _score_terms(self, internal_docids, doc_scores):
        term_scores = {}
        doc_term_data = [] 
        doc_lengths = []
        
        for docid in internal_docids:
            term_tf, length, source = self._cached_document_terms(
                docid, self.exp_field_in
            )
            term_tf = {
                term: frequency for term, frequency in term_tf.items()
                if self._is_valid_term(term)
            }
            doc_term_data.append(term_tf)
            doc_lengths.append(length)

        vocab = set()
        for term_tf in doc_term_data:
            vocab.update(term_tf.keys())

        if internal_docids and not vocab:
            raise RuntimeError(
                "PRF found no terms in field '{}'; the index must provide term vectors "
                "or a stored '{}-string' field.".format(
                    self.exp_field_in, self.exp_field_in
                )
            )

        positive_scores = [max(0.0, float(score)) for score in doc_scores]
        score_total = sum(positive_scores)
        doc_weights = (
            [score / score_total for score in positive_scores]
            if score_total > 0 else [1.0 / len(doc_scores)] * len(doc_scores)
            if doc_scores else []
        )

        for term in vocab:
            if self.algorithm == 'okapi':
                score = self._calculate_okapi_weight(term, doc_term_data)
            elif self.algorithm == 'rm3':
                score = self._calculate_rm3_weight(
                    term, doc_term_data, doc_lengths, doc_weights
                )
            else:
                score = 0.0
                
            if score > 0.0:
                term_scores[term] = score
                
        return term_scores


    def _calculate_okapi_weight(self, term, doc_term_data):
        rdf_t = sum(1 for term_tf in doc_term_data if term in term_tf)
        rsj_weight = RewriteWithPrf._cached_okapi_rsj(self.exp_field_in, term)
        
        # Multiply rdf_t with rsj_weight as requested by the slide formula
        return rdf_t * rsj_weight


    def _calculate_rm3_weight(self, term, doc_term_data, doc_lengths, doc_weights):
        ctf = RewriteWithPrf._cached_total_term_freq(self.exp_field_in, term)
        collection_len = RewriteWithPrf._cached_sum_field_lengths(self.exp_field_in)
        if ctf <= 0 or collection_len <= 0:
            return 0.0
        p_t_c = ctf / float(collection_len)
        rm3_score = 0.0
        for i in range(len(doc_term_data)):
            tf = doc_term_data[i].get(term, 0)
            length = doc_lengths[i]
            if length > 0:
                p_t_d = (tf + self.rm3_mu * p_t_c) / (length + self.rm3_mu)
                rm3_score += p_t_d * doc_weights[i]
        return rm3_score
