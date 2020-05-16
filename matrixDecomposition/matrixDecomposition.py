"""
__authors: Priyanka Punjabi | Komal Sorte
__Topic: Implementation of Different Birkhoff-von-Neumann Decomposition Algorithms
"""
import datetime
import bcolors as bcolors
import numpy as np
from fractions import Fraction
from collections import OrderedDict
import GreedyHeuristic
from sympy import *
import random
import logging

logging.basicConfig(filename='logs.log', level=logging.DEBUG, format='%(levelname)s: %(asctime)s %(message)s')

# Global variables: Formula_terms: map to store all the alpha and permutation matrices Level: Iteration counter for
# keeping a track of the number of times the BvN algorithm runs until it converges and the final decomposition is
# computed.
formula_terms = dict()
level = 0
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


def validateInput(matrix, n, m):
    """
    Check if the number of agents, goods and matrix input are correct.
    :param matrix: n x m input matrix
    :param n: number of agents
    :param m: number of goods
    :return: Boolean value
    """
    return True if len(matrix) >= n and len(matrix) >= m else False


def checkDoublyStochastic(matrix):
    """
    Check if the input matrix is a valid doubly stochastic matrix where the sum of each row and column is the same.
    :param matrix: n x n input matrix
    :return: Boolean value
    """
    col_sum = np.sum(matrix, axis=0)
    row_sum = np.sum(matrix, axis=1)
    return True if len(set(row_sum)) == 1 and len(set(col_sum)) == 1 and row_sum[0] == col_sum[0] else False


def checkBiStochastic(matrix):
    """
    Check if the input matrix is a valid bi-stochastic matrix where the sum of all the rows or columns are the same.
    :param matrix: n x n input matrix
    :return: Booleanv value
    """
    col_sum = np.sum(matrix, axis=0)
    row_sum = np.sum(matrix, axis=1)
    return True if (row_sum == row_sum[0]).all() or (col_sum == col_sum[0]).all() else False


def isNNZ(matrix):
    """
    Check if the matrix contains non - zero elements
    :param matrix: n x n matrix
    :return: Boolean value
    """
    return (matrix == 0).all()


def isPermutationMatrix(matrix):
    """
    This method is used to check where the matrix is a permutation matrix consisiting of only 0's and 1's. If it is a
    permutation matrix, this matrix is stored as the first matrix M that will be used for backtracking.

    :param matrix: n x n matrix
    :return: Boolean Value
    """
    global formula_terms
    global level
    x = np.asanyarray(matrix)
    flag = (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(axis=0) == 1).all() and
            (x.sum(axis=1) == 1).all() and
            ((x == 1) | (x == 0)).all())
    if flag:
        formula_terms["P_" + str(level)] = matrix
        formula_terms["M_" + str(level)] = "P_" + str(level)
    return flag


def findMatching(bipartite):
    """
    This method is based on the Hopcroft-Karp implementation that helps finding maximum matching of the given
    bipartite graph.
    :param bipartite: Map of k, v where k is the agents and its value is the goods to which these agents connect
    :return: maximum/perfect matching of agents to goods
    """
    matching = {}

    # Initial Matching by finding the first possible match
    for agent in bipartite:
        for good in bipartite[agent]:
            if good not in matching:
                matching[good] = agent
                break

    while 1:
        pred_goods = dict()
        unmatched_ele = list()
        pred_agents = dict()

        for agent in bipartite:
            if agent not in pred_agents:
                pred_agents[agent] = -1
            pred_agents[agent] = unmatched_ele

        for ele in matching:
            del pred_agents[matching[ele]]

        layers = list(pred_agents)

        while layers and not unmatched_ele:
            new_layer = dict()

            for agents in layers:
                for goods in bipartite[agents]:
                    if goods not in pred_goods:
                        new_layer.setdefault(goods, []).append(agents)

            layers = []
            for v in new_layer:
                pred_goods[v] = new_layer[v]
                if v in matching:
                    layers.append(matching[v])
                    pred_agents[matching[v]] = v
                else:
                    unmatched_ele.append(v)

        if not unmatched_ele:
            rem_layers = dict()
            for u in bipartite:
                for v in bipartite[u]:
                    if v not in pred_goods:
                        rem_layers[v] = None
            return matching

        # search for alternating paths
        def recurse(ele):
            if ele in pred_goods:
                p_good = pred_goods[ele]
                del pred_goods[ele]

                for g in p_good:
                    if g in pred_agents:
                        pg = pred_agents[g]
                        del pred_agents[g]

                        if pg is unmatched_ele or recurse(pg):
                            matching[ele] = g
                            return 1
            return 0

        for ele in unmatched_ele:
            recurse(ele)


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
        permutation_matrix[v, k] = Fraction(1)

    logging.info('** Permutation Matrix **')
    logging.info(permutation_matrix)

    # Store the permutation matrix for this iteration
    formula_terms["P_" + str(level)] = permutation_matrix

    return permutation_matrix


def computeBipartiteGraph(matrix):
    """
    Compute Bipartite Graph from the given matrix. From this bipartite graph a perfect matching of agents to goods is
    computed. Finally, a permutation matrix of this permutation matrix is returned along with the perfect matching.
    :param matrix: n x m matrix to compute bipartite graph.
    :return: matching and permutation matrix
    """
    global formula_terms
    bipartite = dict()

    # Create a map for storing the connections between agents and goods.
    for ele in range(len(matrix)):
        bipartite[ele] = set()
        temp = np.nonzero(matrix[ele])
        bipartite[ele].update(temp[0])
    # bipartite = OrderedDict(sorted(bipartite.items(), key=lambda i: -len(i[1]), reverse=True))

    logging.info('** Bipartite Graph **')
    logging.info(bipartite)

    # Find a perfect matching from the generated bipartite graph
    matching = findMatching(bipartite)

    logging.info('** Matching **')
    logging.info(matching)

    # Compute a permutation matrix based on the perfect matching
    permutation_matrix = findPermutationMatrix(matrix, matching)

    return matching, permutation_matrix


def findMinimumAlpha(matrix, matching):
    """
    Find minimum alpha by obtaining the minimum value of the matched elements from the matrix
    :param matrix: n x n matrix based on which matching was performed.
    :param matching: map of the matched elements
    :return: minimum alpha value
    """
    global formula_terms
    global level
    current_alpha = -1

    # Iterate over the matched elements to get a minimum value from the matrix
    for k, v in matching.items():
        val = matrix[v, k]
        if current_alpha == -1:
            current_alpha = val
        elif val < current_alpha:
            current_alpha = val

    logging.info('** Minimum Alpha **')
    logging.info(current_alpha)

    # Store the value of minimum alpha obtained in this iteration
    formula_terms["a_" + str(level)] = current_alpha

    return current_alpha


def computeNextFormula(matrix, current_alpha, permutation_matrix, algorithm):
    """
    After getting a permutation matrix, the current matrix is modified by subtracting it with alpha*permutation
    matrix. To normalize the matrix to get a doubly stochastic matrix at the end of each iteration.
    :param matrix: current n x n matrix
    :param current_alpha: the minimum alpha obtained for this iteration
    :param permutation_matrix: the permutation matrix obtained for this iteration
    :param algorithm: the algorithm name that is calling this method i.e., Birkhoff Heuristic or Birkhoff algorithm
    for Bi-stochastic matrices
    :return: modified n x n matrix
    """
    # Subtract matrix with the product of permutation matrix and current alpha
    subtraction_term = abs(np.subtract(matrix, (current_alpha * permutation_matrix)))

    if algorithm == 'Birkhoff':
        # Normalize the resultant matrix to get a doubly stochastic matrix
        matrix = np.array([1 / (1 - current_alpha)] * subtraction_term)
    else:
        matrix = np.array(subtraction_term)

    logging.info('** New Matrix Computed for Next Iteration **')
    logging.info(matrix)

    return matrix


def backwardTracing():
    '''
    Based on the alpha and permutation matrices computed, we compute the desired convex combination of the original
    matrix.
    :return: The final decomposition
    '''
    global level
    global formula_terms
    current = level

    current -= 1

    # Fetch all the alphas and permutation matrices to compute the final decomposition. To do so, first the alpha and
    # permutation matrix obtained at current level and the current+1 matrix M (already a permutation matrix) is
    # considered. (current M) = (current alpha * current permutation matrix) + ((1 - current alpha) * current+1 M)
    while current >= 0:
        alpha = formula_terms["a_" + str(current)]
        subtraction = 1 - alpha
        m_eq = formula_terms["M_" + str(current + 1)]
        m_eval = Mul(float(subtraction), sympify(m_eq))
        formula_terms["M_" + str(current)] = '' + str(float(alpha)) + '*' + "P_" + str(current) + '+' + str(m_eval)
        temp = simplify(formula_terms["M_" + str(current)])
        formula_terms["M_" + str(current)] = temp
        current -= 1

    final_decomposition = formula_terms["M_" + str(current + 1)]
    return final_decomposition


def getFractionalDecomposition(final_decomposition):
    """
    Convert the alphas in the decomposition to a fraction to handle precision problem
    :param final_decomposition:
    :return:
    """
    global formula_terms
    decomp = getTerms(final_decomposition)
    perm_matrix = []
    fractional_decomposition = ''

    for ele in decomp:
        temp = Fraction(ele[0]).limit_denominator()
        perm_matrix.append(formula_terms[ele[1].rstrip()].astype(int))
        fractional_decomposition += str(temp) + '*' + ele[1].rstrip() + ' + '
    fractional_decomposition = fractional_decomposition[:-3]

    return fractional_decomposition, perm_matrix


def validateDecomposition(fractional_decomposition):
    """
    This method is used to validate the generated decomposition by combining each convex combination in the
    decomposition add comparing that with the original matrix to ensure that both the matrices are equal.
    :param fractional_decomposition: Generated decomposition
    :return: Boolean value
    """
    global formula_terms
    global original_matrix

    terms = fractional_decomposition.split(" + ")
    added_decompositions = np.array([])

    for index in range(len(terms)):
        term = terms[index].split("*")
        term[1] = formula_terms[term[1]]
        temp = np.multiply(Fraction(term[0]), term[1])

        if len(added_decompositions) == 0:
            added_decompositions = temp
        else:
            added_decompositions = np.add(added_decompositions, temp)

    return np.array_equal(added_decompositions, original_matrix)


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
    return formula_terms[sample]


def getTerms(decomposition):
    """
    This is a helper method to get individual terms in a combinations from the decomposition
    :param decomposition: generated decomposition of the original matrix
    :return: individual terms in a convex combination
    """
    terms = str(decomposition).split("+")
    decomp = []
    for index in range(len(terms)):
        decomp.append(terms[index].split("*"))

    return decomp


def birkhoff(matrix):
    """
    Given an n x n doubly stochastic matrix, this method runs the Birkhoff Heuristic of the Birkhoff-von-Neumann
    Decomposition to create a convex combination of the given matrix and outputs a deterministic allocation
    for the agents.
    :param matrix: n x n doubly stochastic matrix, row n represents agents and column n represents goods
    :return: deterministic allocation matrix comprising of just 0's and 1's.
    """
    global level
    global formula_terms

    # Iterate until the modified matrix is not a permutation matrix
    while not isPermutationMatrix(matrix):
        # Compute a Bipartite graph of the given input matrix and find a perfect matching of agents to goods. Based
        # on the perfect matching a permutation matrix is derived
        matching, permutation_matrix = computeBipartiteGraph(matrix)

        # Derive the minimum alpha from the computed permutation matrix
        current_alpha = findMinimumAlpha(matrix, matching)

        # Modify the matrix by subtracting permutation matrix with it
        matrix = computeNextFormula(matrix, current_alpha, permutation_matrix, 'Birkhoff')

        level += 1

    # Compute the desired convex combination by backtracking to the alphas and permutation matrices generated above
    final_decomposition = backwardTracing()

    # Convert the alphas into fractions
    fractional_decomposition, perm_matrix = getFractionalDecomposition(final_decomposition)

    # Substituting symbols with their respective Permutation Matrices in the derived decomposition
    decomp = getTerms(fractional_decomposition)
    display_decomp = str(fractional_decomposition)
    for ele in range(len(decomp)):
        display_decomp = display_decomp.replace(decomp[ele][1], str(perm_matrix[ele]))

    logging.info('** Convex Combination **')
    logging.info(display_decomp)

    print('** Convex Combination **')
    print(display_decomp)
    print()

    # Validate the decomposition generated matches the original matrix
    validate_flag = validateDecomposition(fractional_decomposition)
    if validate_flag:
        logging.info('Decomposition Validated')
    else:
        logging.warning('Decomposition Not Validated')

    # Sample a random deterministic allocation from the convex combination
    deterministic_matrix = sampling(fractional_decomposition).astype(int)

    logging.info('Deterministic Solution')
    logging.info(deterministic_matrix)
    return deterministic_matrix


def validateBitochasticDecomposition():
    """
    This method is used to validate the generated decomposition by combining each convex combination in the
    decomposition add comparing that with the original matrix to ensure that both the matrices are equal.
    :return: Boolean value
    """
    global formula_terms
    global original_matrix
    added_decompositions = np.array([])

    for i in range(level):
        decomposition = np.multiply(Fraction(formula_terms['a_' + str(i)]), formula_terms['P_' + str(i)])
        if len(added_decompositions) == 0:
            added_decompositions = decomposition
        else:
            added_decompositions = np.add(added_decompositions, decomposition)

    return np.array_equal(added_decompositions, original_matrix)


def bistochasticSampling():
    """
    This method is used to randomly sample out a convex combination from the decomposition to give a deterministic
    allocation.
    :return: Deterministic solution
    """
    global formula_terms

    randNum = random.randrange(level)
    det_solution = formula_terms['P_' + str(randNum)]

    return det_solution


def biStochasticGetFractionalDecomposition():
    """
    Combine the alpha values and permutation matrices to represent it in the form of a convex combination
    :return: Complete Decomposition
    """
    global formula_terms
    fractional_decomposition = ''

    for i in range(level):
        fractional_decomposition += str(Fraction(formula_terms['a_' + str(i)])) + '*' + str(
            formula_terms['P_' + str(i)].astype(int)) + ' + '

    return fractional_decomposition[:-3]


def birkhoffBistochastic(matrix):
    """
    Given an n x m bi-stochastic matrix, this method runs the Birkhoff Algorithm to create a convex combination of the
    given matrix and outputs a deterministic allocation for the agents.
    :param matrix: n x m bi-stochastic matrix, row n represents agents and column m represents goods
    :return: deterministic allocation matrix comprising of just 0's and 1's.
    """
    global level
    global formula_terms

    # Iterate until the modified matrix does not contain any non-zero entry
    while not isNNZ(matrix):
        # Compute a Bipartite graph of the given input matrix and find a perfect matching of agents to goods. Based
        # on the perfect matching a permutation matrix is derived
        matching, permutation_matrix = computeBipartiteGraph(matrix)

        # Derive the minimum alpha from the computed permutation matrix
        current_alpha = findMinimumAlpha(matrix, matching)

        # Modify the matrix by subtracting permutation matrix with it
        matrix = computeNextFormula(matrix, current_alpha, permutation_matrix, 'bistochastic')

        level += 1

    fractional_decomposition = biStochasticGetFractionalDecomposition()

    # Validate the decomposition generated matches the original matrix
    validate_flag = validateBitochasticDecomposition()

    logging.info('** Convex Combination **')
    logging.info(fractional_decomposition)

    print('** Convex Combination **')
    print(fractional_decomposition)
    print()

    if validate_flag:
        logging.info('Decomposition Validated')
    else:
        logging.warning('Decomposition Not Validated')

    # Sample a random deterministic allocation from the convex combination
    deterministic_matrix = bistochasticSampling().astype(int)

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

def clearGlobalVariables():
    """
    Reset Global Variables
    :return: None
    """
    global original_matrix
    global formula_terms
    global level

    formula_terms = dict()
    level = 0
    original_matrix = []

def main(matrix, heuristic='B', n=0, m=0):
    """
    Based on the n x m input given, this method computes a convex combination of the matrix and samples out a
    deterministic allocation.
    :param matrix: square matrix consisting of allocation of goods to agents
    :param heuristic: Can contain 2 values: Birkhoff / Greedy
    :param n: number of agents
    :param m: number of goods
    :return: deterministic allocation of goods to agents
    """
    global original_matrix

    clearGlobalVariables()

    # Check if the input matrix is in the correct format: type(list)
    if isinstance(matrix, list):
        option = '-1'

        # Convert the elements in input matrix to a fractional form
        matrix = matrix_fractionalPrecomputation(matrix)

        # Check if matrix is a square matrix and the combination of the number of goods and agents match the input
        # matrix
        if checkIfSquare(matrix) and validateInput(matrix, n, m):

            if isPermutationMatrix(matrix):
                logging.info('** Matrix is already a Permutation Matrix **')
                return matrix

            # Check whether the matrix is both a doubly stochastic and bi-stochastic matrix and n = m
            # Accept input for which algorithm to use for decomposition.
            if n == m and checkDoublyStochastic(matrix) and checkBiStochastic(matrix) and n > 0 and m > 0:
                option = input('There are 2 ways to decompose this Input Combination: \nWhich one would you like to '
                               'use? (1 / 2) \n1. Original Birkhoff Heuristic \n2. Birkhoff Algorithm for '
                               'Bi-Stochastic Matrix')

            # Check if matrix is a doubly stochastic matrix with additional checks
            if n == m and checkDoublyStochastic(matrix) and (option == '1' or option == '-1'):
                logging.info('** Matrix is Doubly Stochastic **')

                # Check if it's a 1 x 1 matrix - special case of doubly stochastic matrix
                if len(matrix) == 1:
                    return matrix[0].astype(int)

                # Store original matrix in global variable
                original_matrix = matrix
                logging.info('** Original Input Matrix **')
                logging.info(original_matrix)

                print('** Original Matrix **')
                print(original_matrix)
                print()

                # Check if the parameter heuristic is set to Birkhoff or Greedy
                if heuristic == 'Birkhoff' or heuristic == 'B':
                    decomposition = birkhoff(matrix)
                    return decomposition
                elif heuristic == 'Greedy' or heuristic == 'G':
                    decomposition = GreedyHeuristic.greedy(matrix)
                    return decomposition
                else:
                    logging.error('Invalid Heuristic!')
                    print(bcolors.ERR + 'Invalid Heuristic')
                    return

            # Check if there are any agents or goods to perform decomposition
            elif n > 0 and m > 0:

                # Check if matrix is a bi-stochastic matrix with additional checks
                if (n == m and checkBiStochastic(matrix) and (option == '2' or option == '-1')) or (
                        n != m and checkBiStochastic(matrix)):
                    logging.info('Matrix is Bi-Stochastic')

                    # Check if it's a 1 x 1 matrix - special case of doubly stochastic matrix
                    if len(matrix) == 1:
                        return matrix[0].astype(int)

                    # Store original matrix in global variable
                    original_matrix = matrix
                    logging.info('** Original Input Matrix **')
                    logging.info(original_matrix)

                    print('** Original Matrix **')
                    print(original_matrix)
                    print()

                    deterministic_sol = birkhoffBistochastic(matrix)

                    # Obtain actual decomposition by deleting any dummy objects and assigning each agent the
                    # allocation of its representatives
                    if n != m:
                        print('** Deterministic Solution **')
                        print(deterministic_sol)

                        deterministic_sol = np.delete(deterministic_sol, np.s_[m:], axis=1)
                        for i in range(n):
                            deterministic_sol[i] = np.sum(deterministic_sol[i::n], axis=0)
                        deterministic_sol = np.delete(deterministic_sol, np.s_[n:], axis=0)

                    logging.info('** Final Deterministic Solution **')
                    logging.info(deterministic_sol)

                    return deterministic_sol

                else:
                    print('Matrix is NOT Bi-Stochastic!')
            else:
                print('Insufficient agents or goods for the given matrix!')
                return
        else:
            print('Invalid Input!')
    else:
        print('Incorrect Matrix Input Provided!')


if __name__ == '__main__':
    matrix_doublyStochastic = [
        [1 / 2, 1 / 3, 1 / 6],
        [0, 1 / 6, 5 / 6],
        [1 / 2, 1 / 2, 0]
    ]

    matrix_biStochastic_1 = [
        [0.3, 0.2, 0.2, 0.1],
        [0.2, 0.3, 0.3, 0.4],
        [0.3, 0.2, 0.3, 0.2],
        [0.2, 0.3, 0.2, 0.3]
    ]

    matrix_biStochastic_2 = [
        [1 / 2, 1 / 2, 0, 0, 0, 0],
        [0, 1 / 3, 0, 0, 1 / 3, 1 / 3],
        [1 / 2, 0, 1 / 2, 0, 0, 0],
        [0, 1 / 12, 1 / 4, 0, 1 / 3, 1 / 3],
        [0, 0, 0, 1, 0, 0],
        [0, 1 / 12, 1 / 4, 0, 1 / 3, 1 / 3],
    ]

    matrix_greedy = [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.4, 0.3, 0.2],
        [0.3, 0.2, 0.1, 0.4],
        [0.2, 0.1, 0.4, 0.3]
    ]

    flag = True

    while flag:
        ip = input('Which algorithm do you want to test? \n1. Birkhoff Heuristic for Doubly Stochastic Matrix \n2. '
              'Bistochastic Matrix for n x n Matrix \n3. Bistochastic Matrix for n x m Matrix \n4. Greedy Heuristic for '
              'Doubly Stochastic Matrix')

        starttime = datetime.datetime.now()
        result = []
        if ip == '1':
            print()
            print('** Testing Birkhoff Heuristic on a Doubly Stochastic Matrix **')
            print()
            result = main(matrix_doublyStochastic)

        elif ip == '2':
            print()
            print('** Testing Bistochastic Matrix for n x n Matrix **')
            print()
            result = main(matrix_biStochastic_1, n=4, m=4)

        elif ip == '3':
            print()
            print('** Testing Bistochastic Matrix for n x m Matrix **')
            print()
            result = main(matrix_biStochastic_2, n=3, m=4)

        elif ip == '4':
            print()
            print('** Testing Greedy Heuristic on a Doubly Stochastic Matrix **')
            print()
            result = main(matrix_greedy, 'G')

        print('** Final Deterministic Allocation **')
        print(result)

        endtime = datetime.datetime.now()
        delta = endtime - starttime
        print()
        print(bcolors.OK + 'Time Taken: ', delta)
        print()
        continue_flag = input(bcolors.END + 'Want to test another algorithm? (Y/N)')
        if continue_flag != 'Y' and continue_flag != 'y':
            flag = False
            break