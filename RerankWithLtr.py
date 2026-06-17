"""
A highly modular and hot-pluggable Learning-to-Rank (LTR) reranking manager.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import os
import re
import subprocess
from collections import defaultdict
from functools import cache       # Used for caching static index metrics

import PyLu                       # Interface to the RankLib toolkit
import Util                       # I/O utility functions

from Idx import Idx
from QryParser import QryParser   # Query parsing utilities


# ==============================================================================
# Global Cache for Static Index Background Statistics
# Improves performance by storing frequently accessed, unchanging index data.
# ==============================================================================

@cache
def get_corpus_size():
    """Returns the total number of documents in the index."""
    return float(Idx.getNumDocs())

@cache
def get_field_doc_count(target_field):
    """Returns the number of documents containing the specified field."""
    return float(Idx.getDocCount(target_field))

@cache
def get_corpus_field_length(target_field):
    """Returns the total length of the specified field across the entire collection."""
    return float(Idx.getSumOfFieldLengths(target_field))

@cache
def compute_avg_field_length(target_field):
    """Calculates the average length of a specific field."""
    doc_count = get_field_doc_count(target_field)
    return (get_corpus_field_length(target_field) / doc_count) if doc_count > 0 else 0.0

@cache
def retrieve_doc_freq(target_field, term_str):
    """Fetches the Document Frequency (DF) of a term in a given field."""
    return float(Idx.getDocFreq(target_field, term_str))

@cache
def retrieve_collection_term_freq(target_field, term_str):
    """Fetches the Collection Term Frequency (CTF) of a term."""
    return float(Idx.getTotalTermFreq(target_field, term_str))

@cache
def compute_inverse_doc_freq(target_field, term_str):
    """Computes and caches the Inverse Document Frequency (IDF)."""
    df = retrieve_doc_freq(target_field, term_str)
    if df <= 0:
        return 0.0
    return math.log((get_corpus_size() + 1.0) / (df + 0.5))

@cache
def compute_background_prob(target_field, term_str):
    """Computes the Maximum Likelihood Estimate (MLE) background probability."""
    ctf = retrieve_collection_term_freq(target_field, term_str)
    total_len = get_corpus_field_length(target_field)
    if total_len > 0 and ctf > 0:
        return ctf / total_len
    return 0.0


# ==============================================================================
# Main LTR Reranker Class
# ==============================================================================

class RerankWithLtr:
    """
    Manages the initialization, feature extraction, and prediction pipeline 
    for LTR toolkits (RankLib or SVMRank) using a hot-pluggable architecture.
    """

    def __init__(self, config_params):
        # Base settings
        self.settings = config_params
        self.toolkit_name = str(config_params.get('ltr:toolkit', 'RankLib')).lower()

        # I/O Path definitions
        self.file_train_queries = config_params.get('ltr:trainingQueryFile')
        self.file_train_qrels = config_params.get('ltr:trainingQrelsFile')
        self.file_train_features = config_params.get('ltr:trainingFeatureVectorsFile')
        self.file_test_features = config_params.get('ltr:testingFeatureVectorsFile')
        self.file_test_predictions = config_params.get('ltr:testingDocumentScores')
        self.file_saved_model = config_params.get('ltr:modelFile')
        
        # Pipeline bounds
        self.max_rerank_depth = int(config_params.get('rerankDepth', 100))

        # Toolkit hyperparameters
        self.ranklib_model_id = int(config_params.get('ltr:RankLib:model', 4))
        self.ranklib_opt_metric = config_params.get('ltr:RankLib:metric2t')
        self.svm_learn_exec = config_params.get('ltr:svmRankLearnPath')
        self.svm_classify_exec = config_params.get('ltr:svmRankClassifyPath')
        self.svm_c_value = float(config_params.get('ltr:svmRankParamC', 0.001))

        # Text matching hyperparameters (BM25 & Query Likelihood)
        self.bm25_b_param = float(config_params.get('ltr:BM25:b', 0.75))
        self.bm25_k1_param = float(config_params.get('ltr:BM25:k_1', 1.2))
        self.ql_mu_param = float(config_params.get('ltr:QL:mu', 2500.0))

        # Dynamic feature registry
        self.disabled_feature_ids = self._parse_ignore_list(config_params.get('ltr:featureDisable'))
        self.feature_registry = {}  # Hot-pluggable repository for extractors
        
        # Query-scoped cache to prevent memory explosion
        self.query_tf_buffer = {}

        # Initialization sequence
        self._verify_mandatory_params()
        self._build_output_directories()
        self._bind_default_extractors()
        self._run_training_phase()

    # -------------- Public API for Hot-Pluggability ---------------- #

    def register_feature(self, feat_id, extractor_callable):
        """
        Dynamically injects a new feature extractor into the pipeline.
        
        Args:
            feat_id (int): The integer ID for the feature.
            extractor_callable (callable): A function that accepts exactly three 
                                           arguments: (query_stems, internal_doc_id, is_strict)
                                           and returns a float or None.
        """
        self.feature_registry[feat_id] = extractor_callable

    # -------------- Core Reranking Execution ---------------- #

    def rerank(self, query_batches):
        """
        Re-evaluates and updates the document rankings for a batch of queries 
        using the trained LTR model.
        """
        evaluation_samples = []
        original_mappings = []   # Tracks (query_id, external_id, original_rank)

        for q_identifier, q_info in query_batches.items():
            qid_string = str(q_identifier)
            raw_query_text = str(q_info['qstring'])
            bow_parsed = QryParser.bowQuery(raw_query_text)
            
            # Flatten fielded terms (e.g., from PRF) into pure token streams
            if raw_query_text.strip().startswith('#'):
                bow_parsed = re.sub(r"\b([A-Za-z0-9_]+)\.[A-Za-z0-9_]+\b", r"\1", bow_parsed)
                
            query_tokens = self._extract_stems(bow_parsed)
            self.query_tf_buffer.clear()

            existing_ranking = q_info.get('ranking', [])
            process_bound = min(self.max_rerank_depth, len(existing_ranking))
            
            for rank_pos, (_, ext_doc_id) in enumerate(existing_ranking[:process_bound]):
                try:
                    int_doc_id = Idx.getInternalDocid(ext_doc_id)
                except Exception:
                    continue

                extracted_feats = self._assemble_feature_vector(query_tokens, int_doc_id, strict_mode=False)
                evaluation_samples.append({
                    'qid': qid_string,
                    'label': 0,
                    'external_id': ext_doc_id,
                    'features': extracted_feats
                })
                original_mappings.append((qid_string, ext_doc_id, rank_pos))

        if self.toolkit_name == 'svmrank':
            self._apply_query_level_normalization(evaluation_samples)

        # Synchronize order between feature vectors and their tracking mappings
        aligned_pairs = sorted(zip(evaluation_samples, original_mappings), key=lambda x: self._safe_sort_key(x[0]))
        evaluation_samples = [pair[0] for pair in aligned_pairs]
        original_mappings = [pair[1] for pair in aligned_pairs]

        # Export, Predict, and Load
        self._export_to_disk(evaluation_samples, self.file_test_features)
        self._trigger_prediction_binary()
        predicted_scores = self._parse_prediction_output(self.file_test_predictions)
        
        if len(predicted_scores) != len(original_mappings):
            raise Exception(f'Critical mismatch: Received {len(predicted_scores)} scores for {len(original_mappings)} docs.')

        # Group fresh scores by query identifier
        updated_clusters = defaultdict(list)
        for (qid_string, ext_doc_id, rank_pos), score_val in zip(original_mappings, predicted_scores):
            updated_clusters[qid_string].append((float(score_val), ext_doc_id, rank_pos))

        # Stitch the updated top-K back into the batch records
        for q_identifier in query_batches:
            qid_string = str(q_identifier)
            target_group = updated_clusters.get(qid_string, [])
            
            # Primary sort by score DESC, secondary sort by original rank ASC for stability
            target_group.sort(key=lambda x: (-x[0], x[2]))
            
            reranked_top_tier = [(s, e) for s, e, _ in target_group]
            original_full_ranking = query_batches[q_identifier].get('ranking', [])
            process_bound = min(self.max_rerank_depth, len(original_full_ranking))
            
            orig_head = original_full_ranking[:process_bound]
            orig_tail = original_full_ranking[process_bound:]

            # Retain documents that were in the top-K but failed feature extraction/scoring
            successfully_scored_eids = {e for _, e in reranked_top_tier}
            unscored_fallbacks = [item for item in orig_head if item[1] not in successfully_scored_eids]
            
            query_batches[q_identifier]['ranking'] = reranked_top_tier + unscored_fallbacks + orig_tail
        
        return query_batches

    def _run_training_phase(self):
        """Prepares datasets and invokes the model training engine."""
        raw_train_queries = Util.read_queries(self.file_train_queries) or {}
        qrel_dict = self._compile_qrels(Util.read_qrels(self.file_train_qrels) or [])

        training_dataset = []
        for q_identifier, query_body in raw_train_queries.items():
            qid_string = str(q_identifier)
            if qid_string not in qrel_dict:
                continue
            
            query_tokens = self._extract_stems(query_body)
            self.query_tf_buffer.clear()
            
            for ext_doc_id, relevance_label in qrel_dict[qid_string]:
                try:
                    int_doc_id = Idx.getInternalDocid(ext_doc_id)
                except Exception:
                    continue
                
                features = self._assemble_feature_vector(query_tokens, int_doc_id, strict_mode=False)
                training_dataset.append({
                    'qid': qid_string,
                    'label': int(relevance_label),
                    'external_id': ext_doc_id,
                    'features': features
                })

        if self.toolkit_name == 'svmrank':
            self._apply_query_level_normalization(training_dataset)

        self._export_to_disk(training_dataset, self.file_train_features)
        self._trigger_training_binary()

    # -------------- Feature Assembly Factory ---------------- #

    def _bind_default_extractors(self):
        """
        Populates the dynamic feature registry with the standard 20 features.
        Custom features can be added via the `register_feature` method without 
        altering this base code.
        """
        self.register_feature(1, lambda tk, iid, st: self._feat_spam_indicator(iid))
        self.register_feature(2, lambda tk, iid, st: self._feat_url_slash_count(iid))
        self.register_feature(3, lambda tk, iid, st: self._feat_is_wikipedia(iid))
        self.register_feature(4, lambda tk, iid, st: self._feat_pagerank_val(iid))
        
        self.register_feature(5, lambda tk, iid, st: self._feat_bm25_score(tk, iid, 'body'))
        self.register_feature(6, lambda tk, iid, st: self._feat_ql_score(tk, iid, 'body', st))
        self.register_feature(7, lambda tk, iid, st: self._feat_term_intersection(tk, iid, 'body'))
        
        self.register_feature(8, lambda tk, iid, st: self._feat_bm25_score(tk, iid, 'title'))
        self.register_feature(9, lambda tk, iid, st: self._feat_ql_score(tk, iid, 'title', st))
        self.register_feature(10, lambda tk, iid, st: self._feat_term_intersection(tk, iid, 'title'))
        
        self.register_feature(11, lambda tk, iid, st: self._feat_bm25_score(tk, iid, 'url'))
        self.register_feature(12, lambda tk, iid, st: self._feat_ql_score(tk, iid, 'url', st))
        self.register_feature(13, lambda tk, iid, st: self._feat_term_intersection(tk, iid, 'url'))
        
        self.register_feature(14, lambda tk, iid, st: self._feat_bm25_score(tk, iid, 'inlink'))
        self.register_feature(15, lambda tk, iid, st: self._feat_ql_score(tk, iid, 'inlink', st))
        self.register_feature(16, lambda tk, iid, st: self._feat_term_intersection(tk, iid, 'inlink'))
        
        self.register_feature(17, lambda tk, iid, st: self._feat_body_term_density(tk, iid))
        self.register_feature(18, lambda tk, iid, st: self._feat_max_idf_match(tk, iid))
        self.register_feature(19, lambda tk, iid, st: self._feat_title_coverage(tk, iid))
        self.register_feature(20, lambda tk, iid, st: self._feat_tfidf_sum(tk, iid, 'body'))

    def _assemble_feature_vector(self, query_stems, internal_id, strict_mode=False):
        """Iterates through the registered feature extractors and builds the vector."""
        assembled_vector = {}
        for feat_id, extractor_fn in self.feature_registry.items():
            if feat_id not in self.disabled_feature_ids:
                assembled_vector[feat_id] = extractor_fn(query_stems, internal_id, strict_mode)
        return assembled_vector

    # -------------- Standard Feature Implementations ---------------- #

    def _feat_spam_indicator(self, internal_id):
        return self._convert_to_float(Idx.getAttribute('spamScore', internal_id))

    def _feat_url_slash_count(self, internal_id):
        raw_link = Idx.getAttribute('rawUrl', internal_id)
        return float(raw_link.count('/')) if raw_link else None

    def _feat_is_wikipedia(self, internal_id):
        raw_link = Idx.getAttribute('rawUrl', internal_id)
        return 1.0 if (raw_link and 'wikipedia.org' in raw_link.lower()) else 0.0

    def _feat_pagerank_val(self, internal_id):
        return self._convert_to_float(Idx.getAttribute('PageRank', internal_id))

    def _feat_bm25_score(self, tokens, internal_id, target_field):
        term_dict, _ = self._fetch_term_frequencies(internal_id, target_field)
        if not term_dict: return None
        document_len = Idx.getFieldLength(target_field, internal_id)
        if document_len <= 0: return None
        avg_doc_len = compute_avg_field_length(target_field)
        if avg_doc_len <= 0.0: return None

        accumulated_bm25 = 0.0
        for token in tokens:
            freq = term_dict.get(token, 0)
            if freq > 0 and retrieve_doc_freq(target_field, token) > 0:
                inverse_df = compute_inverse_doc_freq(target_field, token)
                length_penalty = 1.0 - self.bm25_b_param + self.bm25_b_param * (document_len / avg_doc_len)
                denominator = freq + self.bm25_k1_param * length_penalty
                accumulated_bm25 += inverse_df * (freq / denominator)
        return accumulated_bm25

    def _feat_ql_score(self, tokens, internal_id, target_field, strict_mode=False):
        term_dict, _ = self._fetch_term_frequencies(internal_id, target_field)
        if not term_dict: return None
        document_len = Idx.getFieldLength(target_field, internal_id)
        if document_len <= 0: return None
        if not tokens: return 0.0

        matches_found = 0
        running_probability = 1.0
        for token in tokens:
            freq = term_dict.get(token, 0)
            if strict_mode and freq <= 0: return 0.0
            if freq > 0: matches_found += 1
            
            mle_bg_prob = compute_background_prob(target_field, token)
            smoothed_prob = (freq + self.ql_mu_param * mle_bg_prob) / (document_len + self.ql_mu_param)
            if smoothed_prob <= 0.0: return 0.0
            running_probability *= smoothed_prob

        return (running_probability ** (1.0 / len(tokens))) if matches_found > 0 else 0.0

    def _feat_term_intersection(self, tokens, internal_id, target_field):
        term_dict, document_len = self._fetch_term_frequencies(internal_id, target_field)
        if not term_dict or document_len <= 0: return None
        unique_stems = set(tokens)
        return float(sum(1 for t in unique_stems if t in term_dict))

    def _feat_body_term_density(self, tokens, internal_id):
        term_dict, document_len = self._fetch_term_frequencies(internal_id, 'body')
        if not term_dict or document_len <= 0: return None
        if not tokens: return 0.0
        total_tf = sum(term_dict.get(t, 0) for t in tokens)
        return total_tf / float(document_len)

    def _feat_max_idf_match(self, tokens, internal_id):
        term_dict, document_len = self._fetch_term_frequencies(internal_id, 'body')
        if not term_dict or document_len <= 0: return None
        max_idf = 0.0
        for token in set(tokens):
            if token in term_dict and retrieve_doc_freq('body', token) > 0:
                idf = compute_inverse_doc_freq('body', token)
                if idf > max_idf:
                    max_idf = idf
        return max_idf

    def _feat_title_coverage(self, tokens, internal_id):
        term_dict, document_len = self._fetch_term_frequencies(internal_id, 'title')
        if not term_dict or document_len <= 0: return None
        unique_stems = set(tokens)
        if not unique_stems: return 0.0
        successful_hits = sum(1 for tk in unique_stems if tk in term_dict)
        return successful_hits / float(len(unique_stems))

    def _feat_tfidf_sum(self, tokens, internal_id, target_field='body'):
        term_dict, document_len = self._fetch_term_frequencies(internal_id, target_field)
        if not term_dict or document_len <= 0: return None
        tfidf_sum = 0.0
        for token in tokens:
            freq = term_dict.get(token, 0)
            if freq > 0 and retrieve_doc_freq(target_field, token) > 0:
                idf = compute_inverse_doc_freq(target_field, token)
                tfidf_sum += freq * idf
        return tfidf_sum

    # -------------- Executable Invokers ---------------- #

    def _trigger_training_binary(self):
        """Shells out to the external toolkit to compile the training data into a model."""
        if self.toolkit_name == 'ranklib':
            cli_params = [
                '-train', self.file_train_features,
                '-ranker', str(self.ranklib_model_id),
                '-save', self.file_saved_model
            ]
            if self.ranklib_opt_metric:
                cli_params.extend(['-metric2t', str(self.ranklib_opt_metric)])
            elif self.ranklib_model_id == 4:
                cli_params.extend(['-metric2t', 'MAP'])
            PyLu.RankLib.main(cli_params)
            return

        if self.toolkit_name == 'svmrank':
            subprocess.check_output([
                self.svm_learn_exec,
                '-c', str(self.svm_c_value),
                self.file_train_features,
                self.file_saved_model
            ], stderr=subprocess.STDOUT)
            return

        raise Exception('Execution failure: Unsupported toolkit defined as {}'.format(self.toolkit_name))

    def _trigger_prediction_binary(self):
        """Shells out to the external toolkit to predict scores using the generated model."""
        if self.toolkit_name == 'ranklib':
            PyLu.RankLib.main([
                '-rank', self.file_test_features,
                '-load', self.file_saved_model,
                '-score', self.file_test_predictions
            ])
            return

        if self.toolkit_name == 'svmrank':
            subprocess.check_output([
                self.svm_classify_exec,
                self.file_test_features,
                self.file_saved_model,
                self.file_test_predictions
            ], stderr=subprocess.STDOUT)
            return

        raise Exception('Execution failure: Unsupported toolkit defined as {}'.format(self.toolkit_name))

    # -------------- Internal Utilities ---------------- #

    def _fetch_term_frequencies(self, internal_id, target_field):
        """Secures and caches term frequencies safely at the query scope."""
        cache_key = (internal_id, target_field)
        if cache_key in self.query_tf_buffer:
            return self.query_tf_buffer[cache_key]

        vector_data = Idx.getTermVector(internal_id, target_field)
        if not vector_data or vector_data.positionsLength() == 0 or vector_data.stemsLength() == 0:
            self.query_tf_buffer[cache_key] = (None, 0)
            return self.query_tf_buffer[cache_key]

        mapped_frequencies = {}
        for current_idx in range(1, vector_data.stemsLength()):
            stemmed_word = vector_data.stemString(current_idx)
            if stemmed_word:
                mapped_frequencies[str(stemmed_word)] = int(vector_data.stemFreq(current_idx))
                
        self.query_tf_buffer[cache_key] = (mapped_frequencies, int(vector_data.positionsLength()))
        return self.query_tf_buffer[cache_key]

    @staticmethod
    def _safe_sort_key(dictionary_item):
        """Provides a safe multi-type sorting mechanism for query IDs."""
        val = dictionary_item.get('qid')
        try: return (0, int(val))
        except ValueError: pass
        try: return (1, float(val))
        except ValueError: pass
        return (2, str(val))

    def _export_to_disk(self, vector_payloads, target_filepath):
        """Serializes feature vectors into standard RankLib/SVMRank format."""
        file_lines = []
        for sample in sorted(vector_payloads, key=self._safe_sort_key):
            record_fragments = [str(sample['label']), "qid:{}".format(sample['qid'])]
            feat_data = sample['features']
            
            for feat_index in range(1, 21):
                if feat_index in self.disabled_feature_ids: continue
                value = feat_data.get(feat_index)
                if value is None:
                    if self.toolkit_name == 'ranklib':
                        value = 0.0
                    else:
                        continue
                record_fragments.append("{}:{}".format(feat_index, value))
                
            record_fragments.append("# {}".format(sample['external_id']))
            file_lines.append(" ".join(record_fragments))
            
        Util.file_write_strings(target_filepath, file_lines)

    def _parse_prediction_output(self, target_filepath):
        """Reads back the scoring results exported by the toolkit engines."""
        parsed_results = []
        raw_output_lines = Util.file_read_strings(target_filepath) or []
        for line_str in raw_output_lines:
            line_str = line_str.strip()
            if not line_str: continue
            
            tokens = line_str.split()
            if len(tokens) == 1:
                parsed_results.append(float(tokens[0]))
            else:
                try: parsed_results.append(float(tokens[-1]))
                except ValueError: parsed_results.append(float(tokens[0]))
        return parsed_results

    def _apply_query_level_normalization(self, evaluation_dataset):
        """Performs Min-Max scaling on features, typically required by SVMRank."""
        query_partitions = defaultdict(list)
        for document_sample in evaluation_dataset:
            query_partitions[document_sample['qid']].append(document_sample)

        for _, doc_subset in query_partitions.items():
            for feat_index in range(1, 21):
                if feat_index in self.disabled_feature_ids:
                    continue
                extracted_values = [doc['features'].get(feat_index) for doc in doc_subset if doc['features'].get(feat_index) is not None]
                if not extracted_values: continue
                
                floor_val, ceiling_val = min(extracted_values), max(extracted_values)
                if ceiling_val == floor_val:
                    for doc in doc_subset:
                        if doc['features'].get(feat_index) is not None:
                            doc['features'][feat_index] = 0.0
                    continue
                    
                value_range = ceiling_val - floor_val
                for doc in doc_subset:
                    original = doc['features'].get(feat_index)
                    if original is not None:
                        doc['features'][feat_index] = (original - floor_val) / value_range

    def _parse_ignore_list(self, comma_separated_string):
        """Parses a comma-separated string of IDs into a HashSet of integers."""
        omitted = set()
        if not comma_separated_string: return omitted
        for token_part in str(comma_separated_string).split(','):
            token_part = token_part.strip()
            if token_part.isdigit():
                omitted.add(int(token_part))
        return omitted

    def _compile_qrels(self, raw_qrel_matrix):
        """Transforms a raw matrix of Qrel data into a map of judgments."""
        judgment_map = defaultdict(list)
        for data_row in raw_qrel_matrix:
            if len(data_row) < 4: continue
            q_identifier, ext_doc_id, relevance_flag = data_row[0], data_row[2], data_row[3]
            try:
                bounded_relevance = max(0, int(relevance_flag))
                judgment_map[str(q_identifier)].append((ext_doc_id, bounded_relevance))
            except Exception:
                pass
        return judgment_map

    def _convert_to_float(self, raw_input):
        """Safely parses a string or object into a float value."""
        try: return float(raw_input)
        except Exception: return None

    def _extract_stems(self, query_phrase):
        """Tokenizes and stems a raw query phrase."""
        if not query_phrase: return []
        return QryParser.tokenizeString(str(query_phrase))

    def _build_output_directories(self):
        """Ensures that the disk path exists for all generated files."""
        for target_path in [
            self.file_train_features,
            self.file_test_features,
            self.file_test_predictions,
            self.file_saved_model
        ]:
            if not target_path: continue
            parent_dir = os.path.dirname(target_path)
            if parent_dir: os.makedirs(parent_dir, exist_ok=True)

    def _verify_mandatory_params(self):
        """Cross-checks that all required initialization settings have been passed."""
        required_keys = [
            'ltr:trainingQueryFile', 'ltr:trainingQrelsFile',
            'ltr:trainingFeatureVectorsFile', 'ltr:testingFeatureVectorsFile',
            'ltr:testingDocumentScores', 'ltr:modelFile', 'ltr:toolkit'
        ]
        missing_keys = [k for k in required_keys if self.settings.get(k) is None]
        if missing_keys:
            raise Exception('Configuration Block Error: Missing required LTR keys: {}'.format(missing_keys))

        if self.toolkit_name == 'svmrank':
            for k in ['ltr:svmRankLearnPath', 'ltr:svmRankClassifyPath']:
                if self.settings.get(k) is None:
                    raise Exception('Configuration Block Error: Missing SVMRank path key {}'.format(k))
        elif self.toolkit_name == 'ranklib':
            if self.settings.get('ltr:RankLib:model') is None:
                self.ranklib_model_id = 4
        else:
            raise Exception('Configuration Block Error: Unrecognized toolkit {}'.format(self.toolkit_name))