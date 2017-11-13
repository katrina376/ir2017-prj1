import json
import os
import pickle
from gvsm import CorpusVector

def load_DBdoc(origin_path, pickle_path):
    DBdoc_vectors = []

    if not os.path.isfile(pickle_path):
        with open(origin_path, 'r') as jsonfile:
            DBdoc = json.loads(jsonfile.read())

        for idx, doc in enumerate(DBdoc):
            if idx % 1000 == 0:
                print(idx)
            DBdoc_vectors.append(CorpusVector(entity_name=doc['entity'], text=doc['abstract']))

        # Pickle them
        with open(pickle_path, 'wb') as f:
            pickle.dump(DBdoc_vectors, f)
    else:
        with open(pickle_path, 'rb') as f:
            DBdoc_vectors = pickle.load(f)

    return DBdoc_vectors

def load_queries(origin_path, pickle_path):
    query_vectors = []

    if not os.path.isfile(pickle_path):
        with open(origin_path, 'r') as f:
            queries = []
            for line in f:
                strs = line.split('\t')
                queries.append({
                    'id': strs[0].strip(),
                    'query': strs[1].strip(),
                })

        for idx, query in enumerate(queries):
            if idx % 100 == 0:
                print(idx)
            query_vectors.append(CorpusVector(entity_name=query['id'], text=query['query']))

        # Pickle them
        with open(pickle_path, 'wb') as f:
            pickle.dump(query_vectors, f)
    else:
        with open(pickle_path, 'rb') as f:
            query_vectors = pickle.load(f)

    return query_vectors
