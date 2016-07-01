# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: Sobh
"""
import time
import numpy as np
import numpy.linalg as LA
import scipy.sparse as spar
import scipy.io as spio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def form_network_laplacian(network):
    """Forms the laplacian matrix.

    Args:
        network: adjancy matrix.

    Returns:
        diagonal_laplacian: the diagonal of the laplacian matrix.
        laplacian: the laplacian matrix.
    """
    laplacian = network - np.diag(np.diag(network))
    laplacian[laplacian != 0] = 1
    rowsum = sum(laplacian)
    diagonal_laplacian = np.diag(rowsum)

    laplacian = spar.csr_matrix(laplacian)
    diagonal_laplacian = spar.csr_matrix(diagonal_laplacian)

    return diagonal_laplacian, laplacian


def get_a_sample(spreadsheet, percent_sample, lut, network_size):
    """Selects a sample, by precentage, from a given spread sheet.

    Args:
        spreadsheet: genexsample spread sheet.
        percent_sample: percentage of spread sheet to select at random.
        lut: lookup table.
        network_size: network size?? why we need this?

    Returns:
        sample_random: A sample of the spread sheet with the specified precentage.
        patients_permutation: the list the correponds to random sample.
    """
    features_size = np.int_(np.round(spreadsheet.shape[0] * percent_sample))
    features_permutation = np.random.permutation(spreadsheet.shape[0])
    features_permutation = features_permutation[0:features_size]

    patients_size = np.int_(np.round(spreadsheet.shape[1] * percent_sample))
    patients_permutation = np.random.permutation(spreadsheet.shape[1])
    patients_permutation = patients_permutation[0:patients_size]

    sample = spreadsheet[features_permutation.T[:, None], patients_permutation]

    columns = patients_permutation.size

    lut = lut.T
    lut = lut[features_permutation]

    sample_random = np.zeros((network_size, columns))

    for col_id in range(0, columns):
        nanozeros_in_col = (sample[:, col_id] != 0)
        index = lut[nanozeros_in_col]
        index = index[(index >= 0) & (index <= network_size)]
        sample_random[index, col_id] = 1

    #---------------------------
    # eliminate zero columns
    #---------------------------
    positive_col_set = sum(sample_random) > 0
    sample_random = sample_random[:, positive_col_set]
    patients_permutation = patients_permutation[positive_col_set]

    return sample_random, patients_permutation
    
def rwr(restart, network_sparse, alpha=0.7, max_iteration=100, tol=1.e-4,
        report_frequency=5):
    """Performs a random walk with restarts.

    Args:
        restart: restart array of any size.
        network_sparse: adjancy matrix stored in sparse format.
        alpha: restart probability. (default = 0.7)
        max_iteration: maximum number of random walks. (default = 100)
        tol: convergence tolerance. (default = 1.e-4)
        report_frequency: frequency of convergance checks. (default = 5)

    Returns:
        smooth_1: smoothed restart data.
    """
    smooth_0 = restart.copy()
    smooth_r = (1. - alpha) * smooth_0
    smooth_0 = network_sparse.dot(smooth_0) + smooth_r
    for step in range(0, max_iteration):
        smooth_1 = alpha * network_sparse.dot(smooth_0) + smooth_r
        if (step == 1) | (np.mod(step, report_frequency) == 0):
            deltav = LA.norm(smooth_1 - smooth_0, 'fro')
            if deltav < tol:
                break
        smooth_0 = smooth_1
    return smooth_1, step


def quantile_norm(sample):
    """Normalizes an array using quantile normalization (ranking)

    Args:
        sample: initial sample

    Returns:
        sample_quantile_norm: quantile normalized sample
    """
    index = np.argsort(sample, axis=0)           # For each column determine the rank
    sample_sorted_by_rows = np.sort(sample, axis=0)   # Sort each column
    mean_per_row = sample_sorted_by_rows.mean(1) # For each row determine its mean
    sample_quantile_norm = sample.copy()              # initialization
    for j in range(0, sample.shape[1]):
        sample_quantile_norm[index[:, j], j] = mean_per_row[:]
    return sample_quantile_norm


def get_h(w_matrix, x_matrix):
    """Finds a nonnegative right factor (H) of the netnmf function
    X ~ W.H

    Args:
        w_matrix: the positive left factor (W) of the netnmf function
        x_matrix: the postive matrix (X) to be decomposed

    Returns:
        h_matrix: nonnegative right factor (H)
    """
    wtw = np.dot(w_matrix.T, w_matrix)
    number_of_clusters = wtw.shape[0]
    wtx = np.dot(w_matrix.T, x_matrix)
    colix = np.arange(0, x_matrix.shape[1])
    rowix = np.arange(0, w_matrix.shape[1])
    h_matrix = np.dot(LA.pinv(wtw), wtx)
    P = h_matrix > 0
    h_matrix[~P] = 0
    colLogicix = sum(P == 0) > 0
    nC = colix[colLogicix]
    for cluster in range(0, number_of_clusters):
        if nC.size > 0:
            Wk = wtx[:, nC]
            m = Wk.shape[0]
            n = Wk.shape[1]
            ix = np.arange(0, n)
            Hk = np.zeros((m, n))
            Pk = P[:, nC]
            mcoding = np.dot(2**(np.arange(0, m)), np.int_(Pk))
            mcodeU = np.unique(mcoding)
            for k4 in mcodeU:
                ixidx = mcoding == k4
                c = ix[ixidx]
                if c.size > 0:
                    r = rowix[Pk[:, c[0]]]
                    atmp = wtw[r[:, None], r]
                    btmp = Wk[r[:, None], c]
                    A = np.dot(atmp.T, atmp)
                    A = LA.pinv(A)
                    B = np.dot(atmp.T, btmp)
                    Hk[r[:, None], c] = np.dot(A, B)
                    h_matrix[:, nC] = Hk
            P = h_matrix > 0
            h_matrix[~P] = 0
            colLogicix = sum(P == 0) > 0
            nC = colix[colLogicix]
        else:
            break
    return h_matrix


def netnmf(x_matrix, Lk, Ld, k=3, lmbda=1400, itMax=10000, HclustEqLim=200,
           objFcnFreq=50):
    """Performs network based nonnegative matrix factorization that
    minimizes( ||X-WH|| + lambda.tr(W'.L.W)

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W.H
        Lk: the laplacian matrix
        Ld: the diagonal of the laplacian matrix
        k: number of clusters
        lmbda: penalty numnber (default = 100)
        itMax: maximim objective function iterations (default = 10000)
        HclustEqLim: h_matrix no change objective (default = 200)
        objFcnFreq: objective function check interval (default = 50)

    Returns:
        h_matrix: nonnegative right factor (H)
        itr: number of iterations completed
    """
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = np.maximum(w_matrix / np.maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    hClustEQ = np.argmax(h_matrix, 0)
    hEqCount = 0
    for itr in range(0, itMax):
        if np.mod(itr, objFcnFreq) == 0:
            hClust = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(hClustEQ != hClust) == 0):
                hEqCount = hEqCount + objFcnFreq
            else:
                hEqCount = 0
            hClustEQ = hClust
            if hEqCount >= HclustEqLim:
                break
        numerator = np.maximum(np.dot(x_matrix, h_matrix.T)
                               + lmbda * Lk.dot(w_matrix), epsilon)
        denomerator = np.maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T))
                                 + lmbda * Ld.dot(w_matrix), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = np.maximum(w_matrix / np.maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = get_h(w_matrix, x_matrix)
    return h_matrix, itr


def update_connectivity_matrix(H, P, M):
    '''Updates the connectivity matrix

    Args:
        h_matrix: nonnegative right factor (H)
        P: sample permutaion of h_matrix
        M: connectivity matrix

    Returns:
        M: modified connectivity matrix
    '''
    num_clusters = H.shape[0]
    clusterID = np.argmax(H, 0)
    for cluster in range(0, num_clusters):
        sliceID = P[clusterID == cluster]
        M[sliceID[:, None], sliceID] += 1
    return M


def update_indicator_matrix(P, I):
    '''Updates the indicator matrix.

    Args:
        P: sample permutaion of h_matrix
        I: indicator matrix

    Returns:
        I: modified indicator matrix
    '''
    I[P[:, None], P] += 1
    return  I


def reorder_matrix(consensus_matrix, k=3):
    '''Performs K-means and use its labels to reorder the consensus matrix

    Args:
        consensus_matrix: unordered consensus
        k: number of clusters

    Returns:
        M: ordered consensus
    '''
    M = consensus_matrix.copy()
    cluster_handle = KMeans(k, random_state=10)
    labels = cluster_handle.fit_predict(consensus_matrix)
    sorted_labels = np.argsort(labels)
    M = M[sorted_labels[:, None], sorted_labels]
    return M


def get_input():
    '''Gets User's spread sheet and network data

    Args:
        none for now

    Returns:
         network: full gene-gene network
         spreadsheet: user's data
         lut: generated in matlab
    '''
    S = spio.loadmat('testSet.mat')
    network = S["st90norm"]
    spreadsheet = S["sampleMatrix"].T # gene x patient
    lut = S["ucecST90Qlut"]
    lut = np.int64(lut) - 1

    return network, spreadsheet, lut


def echo_input(network, spreadsheet, lut):
    '''Prints User's spread sheet and network data Dimensions and sizes

    Args:
         network: full gene-gene network
         spreadsheet: user's data
         lut: generated in matlab
    '''
    net_rows = network.shape[0]
    net_cols = network.shape[1]
    usr_rows = spreadsheet.shape[0]
    usr_cols = spreadsheet.shape[1]
    date_frm = "Local: %a, %d %b %Y %H:%M:%S"

    print('Data Loaded:\t{}'.format(time.strftime(date_frm, time.localtime())))
    print('adjacency    matrix {} x {}'.format(net_rows, net_cols))
    print('spread sheet matrix {} x {}'.format(usr_rows, usr_cols))
    print('look up table is {}'.format(lut.shape))

    return date_frm


def initialization(spreadsheet):
    '''Initializes connectivity and indicator matrices and setups the run parameters.

    Args:
         network: full gene-gene network
         spreadsheet: user's data

    Returns:
        network_size :
        M :
        I :
    '''
    spSize = spreadsheet.shape[1]
    M = np.zeros((spSize, spSize))
    I = np.zeros((spSize, spSize))

    return  M, I

def parameters_setup():
    '''Setups the run parameters.

    Returns:
        percent_sample :
        number_of_samples :
    '''
    percent_sample = 0.8
    number_of_samples = 5

    return percent_sample, number_of_samples

def display_clusters(M):
    '''Displays the consensus matrix.

    Args:
         M: consenus matrix.
    '''
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    grid = M
    fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    for ax, interp_method in zip(axes.flat, methods):
        ax.imshow(grid, interpolation=interp_method)
        ax.set_title(interp_method)
    plt.show()
    return

