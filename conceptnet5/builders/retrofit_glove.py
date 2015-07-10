import numpy as np
from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import normalize
from conceptnet5.builders.assoc_to_vector_space import negate_concept


class SparseMatrixBuilder:
    """
    SparseMatrixBuilder is a utility class that helps build a matrix of
    unknown shape.
    """

    def __init__(self):
        self.rowIndex = []
        self.colIndex = []
        self.values = []

    def __setitem__(self, key, val):
        row, col = key
        self.rowIndex.append(row)
        self.colIndex.append(col)
        self.values.append(val)

    def tocsr(self, shape, dtype=float):
        return sparse.csr_matrix((self.values, (self.rowIndex, self.colIndex)),
                                shape=shape, dtype=dtype)


def load_conceptnet(filename, labels, verbose=True, offset=1e-9):
    """
    Generates a sparse association matrix from a conceptnet5 csv file.
    """

    mat = SparseMatrixBuilder()

    if verbose:
        print("Loading sparse associations")

    # Add pairwise associations
    totals = defaultdict(float)

    with open(filename, encoding='utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            concept1, concept2, value_str = line.split('\t')
            if concept1 in labels and concept2 in labels:
                index1 = labels.add(concept1)
                index2 = labels.add(concept2)
                value = float(value_str)

                totals[concept1] += value
                totals[concept2] += value
                mat[index1, index2] = value
                mat[index2, index1] = value

    if verbose:
        print("Creating self-loops")

    for index1, concept in enumerate(labels):
        mat[index1, index1] = totals[concept] + 10
        neg = negate_concept(concept)
        if neg in labels:
            index2 = labels.index(neg)
            mat[index1, index2] = -0.5
            mat[index2, index1] = -0.5

    if verbose:
        print("Building sparse matrix")

    return mat.tocsr(shape=(len(labels), len(labels)))


def retrofit(vectors, sparse_matrix, labels, orig_weight=1,
             iterations=20, verbose=True, normalize_intermediate=False):
    """
    Updates the word vectors contained in `dense_file` using the association
    contained in `sparse_file` and writes the new vectors to `output_file`.

    The function will apply retrofitting `iterations` times.

    A larger `offset` causes vectors with small magnitudes to be normalized
    into vectors with magnitudes less than 1.
    """
    orig_vecs = np.copy(vectors)

    if verbose:
        print("Retrofitting")

    for iter in range(iterations):
        if verbose:
            print("%d/%d" % (iter + 1, iterations))

        vectors = sparse_matrix.dot(vectors)

        if normalize_intermediate:
            print("%s normalizing intermediate" % normalize_intermediate)
            normalize(vectors, norm=normalize_intermediate, axis=1, copy=False)

        vectors += orig_vecs * orig_weight
        vectors /= (1 + orig_weight)

        if verbose:
            print("Average diff: %s" % np.mean(np.abs(vectors - orig_vecs)))

    return vectors
