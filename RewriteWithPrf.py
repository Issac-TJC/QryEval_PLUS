"""
Implements Pseudo Relevance Feedback (PRF) for Query Rewriting.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
from functools import cache
from Idx import Idx

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


    def __init__(self, parameters):
        # Default values as specified in the Design Guide
        self.algorithm = parameters.get('prf:algorithm', 'rm3').lower()
        self.num_docs = int(parameters.get('prf:numDocs', 10))
        self.num_terms = int(parameters.get('prf:numTerms', 10))
        self.exp_field_in = parameters.get('prf:expansionFieldIn', 'body')
        self.exp_field_out = parameters.get('prf:expansionFieldOut', 'body')

        # Original weight: if missing, treat as 0
        self.orig_weight = float(parameters.get('prf:rm3:origWeight', 0.0))
        
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
                # WRAP original_qstring IN #SUM() HERE:
                expanded_qstring = f"#WSUM( {self.orig_weight} #SUM({original_qstring}) {new_weight} {learned_query_internal} )"
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
            tv = Idx.getTermVector(docid, self.exp_field_in)
            term_tf = {}
            length = 0
            
            if tv is not None and tv.stemsLength() > 0:
                for stem_i in range(1, tv.stemsLength()):
                    term = tv.stemString(stem_i)
                    if self._is_valid_term(term):
                        term_tf[term] = tv.stemFreq(stem_i)
                
                length = tv.positionsLength()
            
            doc_term_data.append(term_tf)
            doc_lengths.append(length)

        vocab = set()
        for term_tf in doc_term_data:
            vocab.update(term_tf.keys())

        for term in vocab:
            if self.algorithm == 'okapi':
                score = self._calculate_okapi_weight(term, doc_term_data)
            elif self.algorithm == 'rm3':
                score = self._calculate_rm3_weight(term, doc_term_data, doc_lengths, doc_scores)
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


    def _calculate_rm3_weight(self, term, doc_term_data, doc_lengths, doc_scores):
        # Fetch pre-calculated, cached IDF penalty factor
        idf_factor = RewriteWithPrf._cached_rm3_idf_factor(self.exp_field_in, term)
        
        if idf_factor == 0.0:
            return 0.0
            
        rm3_score = 0.0
        for i in range(len(doc_term_data)):
            tf = doc_term_data[i].get(term, 0)
            length = doc_lengths[i]
            
            if tf > 0 and length > 0:
                p_t_d = tf / float(length)
                doc_score = doc_scores[i]
                
                rm3_score += (p_t_d * doc_score * idf_factor)
                
        return rm3_score