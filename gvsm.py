import math
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def multiply(m1, m2):
    """General case of Matrix multiplication."""
    if m1.size == 1 or m2.size == 1:
        return m1 * m2
    else:
        return np.matmul(m1, m2)


class CorpusVector(object):
    stop = set(stopwords.words('english'))

    def __init__(self, entity_name='', text=None):
        self.entity_name = entity_name
        self.text = text.strip('"') if text is not None else ''
        self.tokens = word_tokenize(self.text)

        self.terms = []
        for token in self.tokens:
            if token.lower() in self.stop:
                continue

            if token.isupper() or token.islower():
                self.terms.append(token)
            else:
                self.terms.append(token.lower())

    def get_weights(self, ref_terms, weight_type='', ref_term_counts=[], N=0):
        fs = [self.terms.count(term) for term in ref_terms]
        if weight_type == 'tfidf' and len(ref_terms) == len(ref_term_counts) and N > 0:
            tfs = np.array([1 + math.log2(f) if f > 0 else 0 for f in fs])
            idf = np.array([math.log2(N/n) for n in ref_term_counts])
            return tfs / idf
        else:
            return np.array(fs)

class IndexTermVector(object):
    DBdoc_vectors = None

    def __init__(self, i, index_terms=[], index_term_counts=[], weight_type=''):
        self.i = i
        self.index_terms = index_terms
        self.index_term_counts = index_term_counts

        t = len(index_terms)
        self.minterms = np.array([0, 1])[np.rollaxis(
                np.indices((2,) * t), 0, t + 1)
                .reshape(-1, t)
            ]

        self.termsets = []
        for m in self.minterms:
            self.termsets.append([self.index_terms[idx] for idx, el in enumerate(m) if el == 1])

        self.minterm_vectors = np.zeros((len(self.minterms), len(self.minterms)), int)
        np.fill_diagonal(self.minterm_vectors, 1)

    def get_vector(self):
        """Return the index term vector of the term given by `i`."""
        cirs = [self.calculate_cir(m) for m in self.minterms]

        # Calculate the numerator
        numer = np.zeros((len(self.minterms),))
        for r in range(len(self.minterms)):
            add = self.get_on(self.minterms[r]) * cirs[r] * self.minterm_vectors[r]
            numer = np.add(numer, add)

        # Calculate the denominator
        denom = sum(self.get_on(self.minterms[r]) * cirs[r]**2 for r in range(len(self.minterms)))**0.5

        return numer / denom

    def get_on(self, minterm):
        return minterm[self.i]

    def calculate_cdj(self, doc_vector):
        weights = doc_vector.get_weights(self.index_terms)
        weights[weights != 0] = 1
        return weights

    def calculate_cir(self, minterm):
        index_term = self.index_terms[self.i]

        match_docs = []
        for doc_vector in self.DBdoc_vectors:
            cdj = self.calculate_cdj(doc_vector)
            if np.array_equal(cdj, minterm):
                match_docs.append(doc_vector)

        return sum([doc_vector.terms.count(index_term) for doc_vector in match_docs])
