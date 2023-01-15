import heapq
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import storage
import math
from numpy import dot
from numpy.linalg import norm

import inverted_index_gcp
from inverted_index_gcp import *
# import gensim
# from gensim.models.keyedvectors import KeyedVectors


nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)
ps = PorterStemmer()

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this
                     # many bytes.


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords]

def search_tokenize(text):
    return [(token.group(), 1) for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords]

def read_posting(w, index, base_dir):
    with closing(MultiFileReader()) as reader:
        posting_list = []
        if w in index.posting_locs:
            locs = index.posting_locs[w]
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, base_dir)
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
    return posting_list

def count_words_in_docs(query, index, base_dir):
  docs_counter = Counter()
  for token in np.unique(query):
      posting_list = read_posting(token, index, base_dir)
      for doc in posting_list:
          docs_counter[doc[0]] = docs_counter.get(doc[0], 0) + 1

  return docs_counter

def remove_words_not_in_corpus(tokenized_query, index):
    new_query = []
    for token in tokenized_query:
        if token in index.df.keys():
            new_query.append(token)
    return new_query


class SearchEngine:

    def __init__(self):
        self.anchor_base_dir = '/anchor_index'
        self.title_base_dir = '/title_index'
        self.body_base_dir = '/body_index'
        self.bucket_base_dir = '/home/baronno/314968306_index'

        # self.anchor_index = InvertedIndex.read_index(self.bucket_base_dir + self.anchor_base_dir, 'AnchorIndex')
        self.title_index = InvertedIndex.read_index(self.bucket_base_dir + self.title_base_dir, 'TitleIndex')
        self.body_index = InvertedIndex.read_index(self.bucket_base_dir + self.body_base_dir, 'BodyIndex')


        # self.page_views_dict = InvertedIndex.read_index(self.bucket_base_dir, 'pageviews-202108-user')
        # self.id_to_tf_idf_and_length_dict = InvertedIndex.read_index(self.bucket_base_dir, 'id_to_tf_idf_and_length_dict')
        self.doc_id_to_title_dict = InvertedIndex.read_index(self.bucket_base_dir, 'id_to_title_dict')

        self.body_DL = InvertedIndex.read_index(self.bucket_base_dir, 'DL')
        self.title_DL = InvertedIndex.read_index(self.bucket_base_dir, 'DL_title')

        # self.model = KeyedVectors.load_word2vec_format(self.bucket_base_dir + '/model_wiki.bin', binary=True)

    def query_expansion_word2Vec(self, query, N=3):
        try:
            similar_words = self.model.most_similar(query, topn=N)
            for word, cos_sim in similar_words:
                if cos_sim < 0.75:
                    break
                query.append(word)
        except Exception:
            pass

        return np.unique(query)

    def search(self, query, N=100):
        tokenized_query = tokenize(query)

        # Body Scores
        bm25_body = BM25(self.body_index, self.body_DL, self.bucket_base_dir + self.body_base_dir, 1.7, 0.25)
        body_bm25_scores = bm25_body.search(tokenized_query, 200)

        # Title Scores
        bm25_title = BM25(self.title_index, self.title_DL, self.bucket_base_dir + self.title_base_dir, 1.8, 0.4)
        title_bm25_scores = bm25_title.search(tokenized_query, 200)

        # Merge Scores
        # merged_scores_original_query = self.merge_results(title_bm25_scores, body_bm25_scores, 0.65, 0.35)
        merged_scores = self.merge_results(title_bm25_scores, body_bm25_scores, 0.5, 0.5)

        # Query expansion
        if (len(tokenized_query) < 3):
            expanded_query = self.query_expansion_word2Vec(tokenized_query, 3)
            new_query_body = remove_words_not_in_corpus(expanded_query, self.body_index)
            body_bm25_scores_expanded = bm25_body.search(new_query_body, 300)
            merged_scores = self.merge_results(merged_scores, body_bm25_scores_expanded, 0.5, 0.5)

        else:
            merged_scores = merged_scores


        # new_query_title = remove_words_not_in_corpus(tokenized_query, self.title_index)
        #
        # return sorted(body_bm25_scores.items(), key=lambda x: x[1], reverse=True)[:N]


        # docs_counter_for_title = count_words_in_docs(new_query_title, self.title_index, self.bucket_base_dir + self.title_base_dir)
        # for doc_id in docs_counter_for_title.keys():
        #     docs_counter_for_title[doc_id] = (docs_counter_for_title[doc_id]) / (len(self.doc_id_to_title_dict[doc_id].split()))

        # merged_scores_expanded_query = self.merge_results_dict(title_bm25_scores, title_bm25_scores_expanded, 0.5, 0.5)

        # return [(doc_id, self.doc_id_to_title_dict[doc_id], score) for doc_id, score in merged_scores if doc_id in self.doc_id_to_title_dict]

        return [(doc_id, self.doc_id_to_title_dict[doc_id]) for doc_id, score in merged_scores if doc_id in self.doc_id_to_title_dict][:N]

    def merge_results(self, title_scores, body_scores, title_weight=0.5, body_weight=0.5):
        merged_scores = defaultdict()

        for title_doc_id, title_score in title_scores:
            merged_scores[title_doc_id] = title_weight * title_score

        for body_doc_id, body_score in body_scores:
            if body_doc_id in merged_scores:
                merged_scores[body_doc_id] += body_weight * body_score
            else:
                merged_scores[body_doc_id] = body_weight * body_score

        return sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

    def search_title(self, query):
        tokenized_query = tokenize(query)
        docs_counter = count_words_in_docs(tokenized_query, self.title_index, self.bucket_base_dir + self.title_base_dir)
        return [(doc_id, self.doc_id_to_title_dict[doc_id]) for doc_id, freq in docs_counter.most_common()]

    def search_anchor(self, query):
        anchor_index = InvertedIndex.read_index(self.bucket_base_dir + self.anchor_base_dir, 'AnchorIndex')
        tokenized_query = tokenize(query)
        docs_counter = count_words_in_docs(tokenized_query, anchor_index, self.bucket_base_dir + self.anchor_base_dir)
        return [(doc_id, self.doc_id_to_title_dict[doc_id]) for doc_id, freq in docs_counter.most_common() if doc_id in self.doc_id_to_title_dict]

    def calculate_tf_idf(self, tokenized_query, id_to_tf_idf_and_length_dict):
        query_counter = Counter(tokenized_query)
        epsilon = .0000001

        query_tfidf = {}
        document_tfidf = {}

        query_norm = 0
        for token in query_counter.keys():
            if token in self.body_index.df.keys():
                tf = query_counter[token] / len(tokenized_query)
                df = self.body_index.df[token]
                idf = math.log((len(id_to_tf_idf_and_length_dict)) / (df + epsilon), 10)

                token_tfidf = tf * idf
                query_tfidf[token] = tf * idf
                query_norm += token_tfidf ** 2
                posting = read_posting(token, self.body_index, self.bucket_base_dir + self.body_base_dir)
                for doc_id, frequency in posting:
                    doc_tfidf = idf * frequency / id_to_tf_idf_and_length_dict[doc_id][1]
                    document_tfidf[doc_id] = document_tfidf.get(doc_id, [])
                    document_tfidf[doc_id].append((token, doc_tfidf))

        query_norm = query_norm ** (0.5)

        return query_tfidf, document_tfidf, query_norm

    def calculate_docs_query_similarity(self, query_tfidf, document_tfidf, query_norm, id_to_tf_idf_and_length_dict):
        docs_cosine_similarity = {}
        for doc_id, tokens_tfidf in document_tfidf.items():
            numerator = 0
            docs_cosine_similarity[doc_id] = 0
            for token, tfidf in tokens_tfidf:
                numerator += tfidf * query_tfidf[token]
                denominator = id_to_tf_idf_and_length_dict[doc_id][0] * query_norm
            if denominator != 0:
                docs_cosine_similarity[doc_id] = numerator / denominator

        return docs_cosine_similarity

    def search_body(self, query, N=100):
        id_to_tf_idf_and_length_dict = InvertedIndex.read_index(self.bucket_base_dir, 'id_to_tf_idf_and_length_dict')
        tokenized_query = tokenize(query)
        query_tfidf, document_tfidf, query_norm = self.calculate_tf_idf(tokenized_query, id_to_tf_idf_and_length_dict)
        docs_cosine_similarity = self.calculate_docs_query_similarity(query_tfidf, document_tfidf, query_norm, id_to_tf_idf_and_length_dict)

        return [(doc_id, self.doc_id_to_title_dict[doc_id]) for doc_id, similarity in sorted(docs_cosine_similarity.items(), key=lambda item: item[1], reverse=True) if doc_id in self.doc_id_to_title_dict][:N]

    def get_page_rank(self, list_of_doc_ids):
        page_rank = InvertedIndex.read_index(self.bucket_base_dir, 'pr')
        page_ranks_list = []
        for doc_id in list_of_doc_ids:
            if doc_id in page_rank:
                page_ranks_list.append(page_rank[doc_id])
        return page_ranks_list

    def get_page_views(self, list_of_doc_ids):
        page_views_dict = InvertedIndex.read_index(self.bucket_base_dir, 'pageviews-202108-user')
        page_views_list = []
        for doc_id in list_of_doc_ids:
            if doc_id in page_views_dict:
                page_views_list.append(page_views_dict[doc_id])
        return page_views_list

class BM25:
    def __init__(self, index, DL, base_dir, k1=1.5, b=0.5):
        self.base_dir = base_dir
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values())/self.N
        self.DL = DL

    def calc_idf(self, list_of_tokens):
        idf = {}
        for token in list_of_tokens:
            n_ti = self.index.df[token]
            idf[token] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))

        return idf

    def search(self, query, num_of_docs_to_return):
        idf = self.calc_idf(query)

        scores = {}
        for token in query:
            postings = read_posting(token, self.index, self.base_dir)
            for doc_id, tf in postings:
                scores[doc_id] = scores.get(doc_id, 0) + self._score(token, doc_id, tf, idf)

        return heapq.nlargest(num_of_docs_to_return, [(doc_id,score) for doc_id, score in scores.items()], key=lambda x: x[1])

    def _score(self, token, doc_id, doc_tf, idf):
        numerator = idf[token] * doc_tf * (self.k1 + 1)
        denominator = doc_tf + self.k1 * (1 - self.b + self.b * self.DL[doc_id] / self.AVGDL)
        return (numerator / denominator)
