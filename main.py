from gvsm import *
from load import *

import numpy as np

SAVE_NUM = 50
LIST_NUM = 5

try:
    os.remove('result_file.txt')
except OSError:
    pass

print('Load DBdoc_vectors...')
DBdoc_vectors = load_DBdoc('DBdoc.json', 'DBdoc_vectors.pickle')
IndexTermVector.DBdoc_vectors = DBdoc_vectors

print('Load query_vectors...')
query_vectors = load_queries('queries-v2.txt', 'query_vectors.pickle')

print('Set up')

for idx, query_vec in enumerate(query_vectors):
    print('=== Query:', query_vec.text)
    index_terms = query_vec.terms

    print('=== Start calculate index term vector')
    index_term_vectors = np.zeros((1, len(index_terms)))
    for i, term in enumerate(index_terms):
        index_term_vec = IndexTermVector(i, index_terms=index_terms)
        print(i, term)
        if i == 0:
            index_term_vectors = index_term_vec.get_vector()
        else:
            index_term_vectors = np.vstack([index_term_vectors, index_term_vec.get_vector()])

    print('=== Translate query into minterm vectors form')
    query_vec_minterm = np.matmul(query_vec.get_weights(index_terms), index_term_vectors)

    print('=== Translate DBdoc into minterm vectors form')
    scores = []
    for DBdoc_vec in DBdoc_vectors:
        DBdoc_vec_minterm = np.matmul(DBdoc_vec.get_weights(index_terms), index_term_vectors)

        numer = np.dot(DBdoc_vec_minterm, query_vec_minterm.T)
        denom = np.linalg.norm(DBdoc_vec_minterm) * np.linalg.norm(query_vec_minterm)

        score = numer/denom if denom != 0 else 0
        scores.append({
            'entity': DBdoc_vec.entity_name,
            'score': score,
            'ranking': 0
        })

    print('=== Top', LIST_NUM, 'result of Query:', query_vec.text)
    rankings = sorted(scores, key=lambda k: k['score'], reverse=True)
    for idx, rank in enumerate(rankings):
        rank['ranking'] = idx + 1

    print(' '.join([rank['entity'] for rank in rankings if rank['ranking'] < LIST_NUM + 1]))

    with open('result_file.txt', 'a', encoding='UTF-8') as out:
        for rank in (rank for rank in rankings if rank['ranking'] < SAVE_NUM + 1):
            out.write('\t'.join([
                    query_vec.entity_name,                 # query_ID
                    'Q0',                                  # Q0
                    '<dbpedia:{}>'.format(rank['entity']), # <dbpedia:entity>
                    str(rank['ranking']),                       # ranking
                    str(rank['score']),                         # score
                    'STANDARD'                             # STANDARD
                ]) + '\n')
