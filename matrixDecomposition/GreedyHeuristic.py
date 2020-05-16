"""
__authors: Priyanka Punjabi | Komal Sorte
__Topic: Implementation of Birkhoff-von-Neumann Decomposition (Greedy Heuristic)
"""

import numpy as np
from fractions import Fraction
from collections import OrderedDict
from sympy import *
import random
import math
import logging

logging.basicConfig(filename='logs.log', level=logging.DEBUG, format='%(levelname)s: %(asctime)s %(message)s')

formula_terms = dict()
level = 0
k = 0
bottleNeck = 0
alpha_dict = dict()
alpha_count = -1
original_matrix = []


def checkIfSquare(matrix):
    """
    Check if the input matrix is a square matrix
    :param matrix: n x n input matrix
    :return: Boolean value
    """
    if len(matrix.shape) == 2:
        return True if matrix.shape[0] == matrix.shape[1] else False
    else:
        return False


def checkDoublyStochastic(matrix):
    """
    Check if the input matrix is a valid doubly stochastic matrix where the sum of each row and column is the same.
    :param matrix: n x n input matrix
    :return: Boolean value
    """
    col_sum = np.sum(matrix, axis=0)
    row_sum = np.sum(matrix, axis=1)
    return True if len(set(row_sum)) == 1 and len(set(col_sum)) == 1 and row_sum[0] == col_sum[0] else False


def isNNZ(matrix):
    """
    Check if the matrix contains non - zero elements
    :param matrix: n x n matrix
    :return: Boolean value
    """
    return (matrix == 0).all()


def alterMatrix(matrix, permutation_matrix):
    """
    After getting a permutation matrix, the current matrix is modified by subtracting it with alpha*permutation
    matrix. To normalize the matrix to get a doubly stochastic matrix at the end of each iteration.
    :param matrix: current n x n matrix
    :param permutation_matrix: the permutation matrix obtained for this iteration
    :return: modified n x n doubly stochastic matrix
    """
    global alpha_dict
    global alpha_count

    # Subtract matrix with the product of permutation matrix and current alpha
    matrix = np.array(abs(np.subtract(matrix, (alpha_dict["a_" + str(alpha_count)] * permutation_matrix))))

    logging.info('** New Matrix Computed for Next Iteration **')
    logging.info(matrix)

    return matrix


def findPermutationMatrix(matrix, matching):
    """
    Based on the matching obtained for a matrix compute a permutation matrix by setting 1 for the matched elements
    and 0 for the remaining.
    :param matrix: current n x n matrix
    :param matching: perfect matching computed based on bipartite graph
    :return: permutation matrix for the given matching with elements consisting of only 0s and 1s
    """
    global formula_terms
    global level
    permutation_matrix = np.full(matrix.shape, Fraction(0, 1))

    # Iterate over each matching element and assign 1 to those elements in the permutation matrix
    for k, v in matching.items():
        permutation_matrix[k, v] = Fraction(1)

    logging.info('** Permutation Matrix **')
    logging.info(permutation_matrix)

    # Store the permutation matrix for this iteration
    formula_terms["P_" + str(level)] = permutation_matrix

    return permutation_matrix


def extremeMatching(matrix, bipartite_col):
    """
    Perform intial extreme matching before checking for augmenting paths.
    :param matrix: Updated matrix
    :param bipartite_col: Bipartite graph which is computed for current iteration
    :return: Returns a dictionary of matched nodes and a list of unmatched nodes.
    """
    global bottleNeck
    global alpha_count
    matching = dict()
    unmatched = list()

    row_min = np.where(matrix > 0, matrix, matrix.max()).min(axis=1)
    col_min = np.where(matrix > 0, matrix, matrix.max()).min(axis=0)

    # compute bottleneck value
    bottleNeck = Fraction(max(max(row_min), max(col_min)))
    logging.info("** BottleNeck **")
    logging.info(bottleNeck)

    alpha_count += 1
    if len(alpha_dict) == 0:
        alpha_dict["a_" + str(alpha_count)] = bottleNeck
    else:
        alpha_dict["a_" + str(alpha_count)] = min(min(alpha_dict.values()), bottleNeck)
    for col, rows in bipartite_col.items():
        flag = False
        for row in rows:
            if matrix[row][col] <= bottleNeck and row not in matching:
                matching[row] = col
                flag = True
                break
        if not flag:
            unmatched.append(col)

    unmatched = set(unmatched)

    temp = set()
    for col in unmatched:
        for row in bipartite_col[col]:
            if row in matching and matrix[row][col] <= bottleNeck:
                j1 = matching[row]
                for j1_row in bipartite_col[j1]:
                    if j1_row not in matching:
                        matching[j1_row] = j1
                        matching[row] = col
                        temp.add(col)

    unmatched = unmatched.difference(temp)

    logging.info("** Matching **")
    logging.info(matching)
    logging.info("** Unmatched **")
    logging.info(unmatched)

    return matching, unmatched


def checkAugmentingPaths(matrix, j0, matching, bipartite_col):
    """
    Checks for a possible augmented path and returns an augmented path if found.
    :param matrix: Updated matrix
    :param j0: Unmatched column
    :param matching: Extreme matching computed so far in the iteration
    :param bipartite_col: Bipartite graph which is computed for current iteration
    :return: Possible augmenting path
    """
    global bottleNeck

    # Contains (marked) vertices whose distances to node j0 and shortest alternating path are known.
    B = set()

    # Contains vertices for which an alternating path to the root is known. This alternating path may or may not be
    # the shortest possible.
    Q = set()

    # Probabilistic allocation.
    d = dict()
    for ele in range(len(matrix)):
        if ele not in d:
            d[ele] = math.inf

    # Length of shortest path from j0 to any node in Q
    lsp = 0

    # Length of shortest augmenting path
    lsap = math.inf
    j = j0
    path = set()
    isap = -1
    b = bottleNeck
    prev = -1

    while 1:
        rows = bipartite_col[j]
        col_j = rows.difference(B)
        if len(col_j) == 0:
            temp = set()
            for ele in path:
                if ele[0] == prev:
                    temp.add(ele)
            path = path.difference(temp)
        for i in col_j:
            dnew = max(lsp, matrix[i][j])
            if dnew < lsap:
                if i not in matching:
                    lsap = dnew
                    isap = i
                    path.add((isap, j))
                    if lsap <= b:
                        return
                else:
                    if dnew < d[i]:
                        d[i] = dnew
                        path.add((i, j))
                        if i not in Q:
                            Q.add(i)
        # Exit is Q is empty.
        if len(Q) == 0:
            break
        min_temp = math.inf
        i = -1
        for row in Q:
            if d[row] < min_temp:
                min_temp = d[row]
                i = row
        lsp = min_temp
        if lsap <= lsp:
            break
        Q.remove(i)
        B.add(i)
        j = matching[i]
        prev = i
        # Possible augmenting path from node isap to node j0;
        path.add((i, j))
    if lsap != math.inf:
        logging.info('** Augment **')
        logging.info(path)
        return path


def computeBipartiteGraph(matrix):
    """
    Compute Bipartite Graph from the given matrix. From this bipartite graph a perfect matching of agents to goods is
    computed. Finally, a permutation matrix of this permutation matrix is returned along with the perfect matching.
    :param matrix: n x m matrix to compute bipartite graph.
    :return: matching and permutation matrix
    """
    global formula_terms
    bipartite_row = dict()
    bipartite_col = dict()
    nonzero_array = np.transpose(np.nonzero(matrix))

    # Create a map for storing the connections between agents and goods.
    for ele in range(len(nonzero_array)):
        if nonzero_array[ele][0] not in bipartite_row:
            bipartite_row[nonzero_array[ele][0]] = set()

        if nonzero_array[ele][1] not in bipartite_col:
            bipartite_col[nonzero_array[ele][1]] = set()

        bipartite_row[nonzero_array[ele][0]].add(nonzero_array[ele][1])
        bipartite_col[nonzero_array[ele][1]].add(nonzero_array[ele][0])

    logging.info('** Bipartite Graph **')
    logging.info(bipartite_row)
    logging.info(bipartite_col)

    # Find a perfect matching from the generated bipartite graph
    # First, perform extreme matching
    matching, unmatched = extremeMatching(matrix, bipartite_col)
    # if the there are any unmatched vertices then check for an augmenting path and implement it.
    for col in unmatched:

        # chech for an aurgmenting path
        path = checkAugmentingPaths(matrix, col, matching, bipartite_col)
        temp = set()

        # process the augmenting path and update the current matching accordingly.
        for k, v in matching.items():
            if (k, v) in path:
                path.remove((k, v))
                temp.add(k)
        for ele in temp:
            del matching[ele]
        for ele in path:
            matching[ele[0]] = ele[1]

    logging.info('** Matching **')
    logging.info(matching)

    # Compute a permutation matrix based on the perfect matching
    permutation_matrix = findPermutationMatrix(matrix, matching)

    return matching, permutation_matrix


def getFractionalDecomposition(final_decomposition):
    """
    Convert the alphas in the decomposition to a fraction to handle precision problem
    :param final_decomposition:
    :return:
    """
    terms = str(final_decomposition).split("+")
    fractional_decomposition = ''
    for index in range(len(terms)):
        term = terms[index].split("*")
        temp = Fraction(term[0]).limit_denominator()
        fractional_decomposition += str(temp) + '*' + term[1].rstrip() + ' + '
    fractional_decomposition = fractional_decomposition[:-3]

    return fractional_decomposition


def validateDecomposition():
    """
    This method is used to validate the generated decomposition by combining each convex combination in the
    decomposition add comparing that with the original matrix to ensure that both the matrices are equal.
    :return: Boolean value
    """
    global formula_terms
    global alpha_dict
    global original_matrix
    added_decompositions = np.array([])
    for item in range(len(alpha_dict)):
        temp = np.multiply(formula_terms['P_' + str(item)], alpha_dict['a_' + str(item)])
        if len(added_decompositions) == 0:
            added_decompositions = temp
        else:
            added_decompositions = np.add(added_decompositions, temp)
    return np.array_equal(added_decompositions, original_matrix)


def getDecomposition():
    '''
    Based on the alpha and permutation matrices computed, we compute the desired convex combination of the original
    matrix.
    :return: The final decomposition
    '''
    global formula_terms
    global alpha_dict
    final_decomposition = ''

    # Fetch all the alphas and permutation matrices to compute the final decomposition and
    # then simply combine them all in a format -> [alpha_j * P_j] + ... + [alpha_k * P_k]
    for item in range(len(alpha_dict)):
        temp = np.array(formula_terms['P_' + str(item)]).astype(int)
        final_decomposition += str(alpha_dict['a_' + str(item)]) + '*' + str(temp) + ' + '

    final_decomposition = final_decomposition[:-3]

    return final_decomposition


def sampling(fractional_decomposition):
    """
    This method is used to randomly sample out a convex combination from the decomposition to give a deterministic
    allocation.
    :param fractional_decomposition: generated decomposition of the original matrix
    :return: Deterministic solution
    """
    global formula_terms
    samples = fractional_decomposition.split(' + ')
    max_len = len(samples)
    randNum = random.randrange(max_len)
    sample = samples[randNum].split('*')[1]
    return sample


def greedy(matrix):
    """
    Given an n x n doubly stochastic matrix, this method runs the Greedy Heuristic of the Birkhoff-von-Neumann
    Decomposition to create a convex combination of the given matrix and output a deterministic allocation for each
    agent.
    :param matrix: n x n doubly stochastic matrix, row n represents agents and column n represents goods
    :return: deterministic allocation matrix comprising of just 0's and 1's.
    """
    global level
    global formula_terms
    global k

    # Iterate until the modified matrix contains non-zero elements.
    while not isNNZ(matrix):
        k += 1

        # Compute a Bipartite graph of the given input matrix and find a perfect matching of agents to goods. Based
        # on the perfect matching a permutation matrix is derived
        matching, permutation_matrix = computeBipartiteGraph(matrix)

        # Modify the matrix by subtracting permutation matrix with it
        matrix = alterMatrix(matrix, permutation_matrix)

        level += 1

    # Compute the final convex combination by putting together all the alphas and permutation matrices obtained
    # up til this point.
    fractional_decomposition = getDecomposition()

    logging.info('** Convex Combination **')
    logging.info(fractional_decomposition)
    print('** Convex Combination **')
    print(fractional_decomposition)

    # Validate the decomposition generated matches the original matrix
    validate_flag = validateDecomposition()
    if validate_flag:
        logging.info('** Decomposition Validated **')
    else:
        logging.warning('** Decomposition Not Validated **')

    # Sample a random deterministic allocation from the convex combination
    deterministic_matrix = sampling(fractional_decomposition)

    logging.info('** Deterministic Solution **')
    logging.info(deterministic_matrix)

    return deterministic_matrix


def matrix_fractionalPrecomputation(matrix):
    """
    Convert each element in the matrix to a fraction to handle precision preoblem
    :param matrix: n x m input matrix
    :return: matrix with fractional elements
    """
    row = -1
    for ele in matrix:
        row += 1
        col = -1
        for e in ele:
            col += 1
            matrix[row][col] = Fraction(e).limit_denominator()
    return np.array(matrix)