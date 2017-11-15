from gvsm import *
from load import *

import numpy as np

SAVE_NUM = 1000
LIST_NUM = 5

RESULT_FILE_NAME = 'result_file.txt'

try:
    os.remove(RESULT_FILE_NAME)
except OSError:
    pass

print('Load DBdoc_vectors...')
DBdoc_vectors = load_DBdoc('DBdoc.json', 'DBdoc_vectors.pickle')
IndexTermVector.DBdoc_vectors = DBdoc_vectors

print('Load query_vectors...')
query_vectors = load_queries('queries-v2.txt', 'query_vectors.pickle')

for idx, query_vec in enumerate(query_vectors):
    print('=== Query:', query_vec.text)
    index_terms = query_vec.terms

    # Count the appearance of each index terms for implementing TF-IDF
    index_term_counts = []
    for term in index_terms:
        count = [(term in DBdoc_vec.terms) for DBdoc_vec in DBdoc_vectors].count(True)
        index_term_counts.append(count)

    print('====== Calculate index term vector')
    # Go by each single index term, achieve the stack of all index term vectors
    index_term_vectors = np.zeros((1, len(index_terms)))
    for i, term in enumerate(index_terms):
        print('\t', i, term)

        # Create the index term object
        # More information of calculating can be found in gvsm.py
        index_term_vec = IndexTermVector(i, index_terms, index_term_counts)

        # Calculate the vector; if it is not the first term, put below the stack
        if i == 0:
            index_term_vectors = index_term_vec.get_vector()
        else:
            index_term_vectors = np.vstack([index_term_vectors, index_term_vec.get_vector()])

    print('====== Translate query into minterm vectors form')
    # Translate the query vector into minterm vector space
    query_vec_minterm = multiply(
            query_vec.get_weights(
                    ref_terms=index_terms,
                    weight_type='tfidf',
                    ref_term_counts=index_term_counts,
                    N=len(DBdoc_vectors)
                ),
            index_term_vectors
        )

    print('====== Translate DBdoc into minterm vectors form')
    # Go by each single DBdoc, calculate the score
    scores = []
    for DBdoc_vec in DBdoc_vectors:
        # Translate the DBdoc vectors into minterm vector space
        DBdoc_vec_minterm = multiply(
                DBdoc_vec.get_weights(
                        ref_terms=index_terms,
                        weight_type='tfidf',
                        ref_term_counts=index_term_counts,
                        N=len(DBdoc_vectors)
                    ),
                index_term_vectors
            )

        # Cosine similarity is used for scoring
        numer = np.dot(DBdoc_vec_minterm, query_vec_minterm.T)
        denom = np.linalg.norm(DBdoc_vec_minterm) * np.linalg.norm(query_vec_minterm)

        score = numer / denom if denom != 0 else 0
        scores.append({
            'entity': DBdoc_vec.entity_name,
            'score': score,
            'ranking': 0
        })

    print('====== Top', LIST_NUM, 'result of Query:', query_vec.text)
    # Sort the list by score
    rankings = sorted(scores, key=lambda k: k['score'], reverse=True)
    for idx, rank in enumerate(rankings):
        rank['ranking'] = idx + 1

    print('\t', '; '.join([rank['entity'] for rank in rankings if rank['ranking'] < LIST_NUM + 1]))

    # Save the top 50 results
    with open(RESULT_FILE_NAME, 'a', encoding='UTF-8') as out:
        for rank in (rank for rank in rankings if rank['ranking'] < SAVE_NUM + 1):
            out.write('\t'.join([
                    query_vec.entity_name,                 # query_ID
                    'Q0',                                  # Q0
                    '<dbpedia:{}>'.format(rank['entity']), # <dbpedia:entity>
                    str(rank['ranking']),                  # ranking
                    str(rank['score']),                    # score
                    'STANDARD'                             # STANDARD
                ]) + '\n')
