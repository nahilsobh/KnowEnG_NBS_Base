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
import h5py
import argparse

def consensus_cluster_nbs(network_sparse, spreadsheet, Ld, Lk, nbs_par_set):
    """ main loop for this module computes the components for the consensus matrix
        from the input network and spreadsheet
    Args:
        network: genes x genes symmetric adjacency matrix
        spreadsheet: genes x samples matrix
        Ld, Lk: laplacian matrix components i.e. L = Ld - Lk
        nbs_par_set = {"number_of_bootstraps":1, "percent_sample":0.8, "k":3,
                    "rwr_alpha":0.7}
        number_of_samples: number of iterations of nbs to try
        percent_sample: portion of spreadsheet to use in each iteration
        k: inner dimension of matrx factorization
        alpha: radom walk with restart proportions
    
    Returns:
        connectivity_matrix: samples x samples count of sample relations
        indicator_matrix: samples x samples count of sample trials
    """
    number_of_samples = nbs_par_set["number_of_bootstraps"]
    percent_sample = nbs_par_set["percent_sample"]
    k = nbs_par_set["k"]
    alpha = nbs_par_set["rwr_alpha"]
    connectivity_matrix, indicator_matrix = initialization(spreadsheet)

    # ----------------------------------------------
    # Network based clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, number_of_samples):
        sample_random, sample_permutation = spreadsheet_sample(spreadsheet, percent_sample)
        sample_smooth, iterations = rwr(sample_random, network_sparse, alpha)
        print("iterations = ", iterations)
        sample_quantile_norm = quantile_norm(sample_smooth)
        H, niter = netnmf(sample_quantile_norm, Lk, Ld, k)
        connectivity_matrix = update_connectivity_matrix(H, sample_permutation, connectivity_matrix)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)
        
    return connectivity_matrix, indicator_matrix
    
def data_frame_to_3col_numeric(df_input):
    """ convert pandas data frame (gene, gene, wt., type) to 3 column float
    Args:
        df_input: pandas (4 column) data frame  with ensembel names in col 0, 1
    Returns:
        numeric_3col_mat: 3 x number of valid ENSG... gene name pairs matrix
    """
    V = df_input.values
    good_row_list = np.int32(np.zeros((V.shape[0])))
    good_string = 'ENS'
    n = 0
    for row in range(0, V.shape[0]):
        a = V[row, 0]
        b = V[row, 1]
        if not((a[0:3] != good_string) | (b[0:3] != good_string)):
            good_row_list[n] = row
            n += 1
            
    good_row_list = good_row_list[0:n-1]
    V = V[good_row_list, :]
    
    numeric_3col_mat = np.zeros((V.shape[0], 3))
    for row in range(0, V.shape[0]):
        c0 = V[row, 0]
        numeric_3col_mat[row, 0] = float(c0[4:15])
        c1 = V[row, 1]
        numeric_3col_mat[row, 1] = float(c1[4:15])
        numeric_3col_mat[row, 2] = V[row, 2]
    
    return numeric_3col_mat
    
# trim node_links_data and get the unique node names
def nodes_to_matrix(node_links_all, threshold, data_column=2):
    """ Construct an adjacency matrix and it's list of row and column labels from
        a spreadsheet of nodes and links [node, node, w1, w2, w3,...] x times
    Args:
        node_links_all, threshold, data_column
    Returns:
        adjacency_matrix, node_names
    Raises:
        No exceptions are raised internally.
    """
    node_links = node_links_all.copy()
    node_links[:, 0] = node_links[:, 0] - 1
    node_links[:, 1] = node_links[:, 1] - 1
    rowix = np.flipud(np.argsort(node_links[:, data_column]))
    node_links = node_links[rowix, :]
    node_links = node_links[1:np.int_(np.ceil(node_links.shape[0] * threshold) + 1), :]
    node_names = np.unique(np.concatenate(np.array([node_links[:, 0], node_links[:, 1]])))
    matrix_length = node_names.size
    node_index = np.arange(0, matrix_length)
    rows_length = node_links.shape[0]
    adjacency_triples = np.zeros((rows_length * 2, 3))
    for m in range(0, rows_length):
        mm = m + rows_length
        adjacency_triples[m, 0] = node_index[node_names == node_links[m, 0]]
        adjacency_triples[mm, 1] = adjacency_triples[m, 0]
        adjacency_triples[m, 1] = node_index[node_names == node_links[m, 1]]
        adjacency_triples[mm, 0] = adjacency_triples[m, 1]
        adjacency_triples[m, 2] = node_links[m, data_column]
        adjacency_triples[mm, 2] = adjacency_triples[m, 2]
    adjacency_matrix = spar.csr_matrix((
        adjacency_triples[:, 2], (adjacency_triples[:, 0], adjacency_triples[:, 1])),
        shape=(matrix_length, matrix_length) )
    
    return adjacency_matrix, node_names
    
def is_member(a_array, b_array):
    """ Find existance and locations of array "a" in another array "b", when any element of "a"
        occurs more than once in "b" only the first location is retained in the a_index array.
    Args:
        a_array : an array of real numbers.
        b_array : an array of real numbers.
    Returns:
        a_in_b  (logical): true when element of a is found in b.
        a_index     (int): location where each element of a is found in b, or -1 if not found.
    Raises:
        No exceptions are raised internally.
    """
    size_of_a = a_array.size
    size_of_b = b_array.size
    
    b_index = np.arange(0, size_of_b)
    a_in_b = np.zeros(size_of_a,dtype=bool)
    a_index = np.int_(np.zeros(size_of_a) - 1)
    
    for element in range(0, size_of_a):
        element_index = b_index[a_array[element] == b_array]
        if element_index.size >= 1:
            a_in_b[element] = True
            a_index[element] = element_index[0]
            
    return a_in_b, a_index
    
def hpf5_ensemble_dataset_to_int64_array(ensg_dataset):
    """ for hdf5 file fixed length uint8 string
    Args:
        ensg_dataset: type h5py data set such as - ensg_dataset = f['keyStrings']
        whth all same type ensembel strings (ENSG00000000000). Note that the 
        h5 file must be open and available in the namespace
    Returns:
        int64_array: integer array of the numerical part of ensembel strings
    """
    with ensg_dataset.astype('uint8'):
        keys = ensg_dataset[:]
    int64_array = np.int64(np.zeros(keys.shape[1]))
    key_strings = np.string_(np.reshape(keys, (1, keys.size)))
    key_strings = key_strings.splitlines()
    key_number = 0
    for key in key_strings:
        int64_array[key_number] = np.int64(float(key[4:15]))
        key_number += 1
        
    return int64_array
    
def pandas_index_ensg_to_int64_array(row_names):
    """ for pandas index array
    Args:
        row_names: a pandas dataframe index field of ensembel gene names
        
    Returns:
        int64_array: 64 bit integer array equ
    """
    rows = row_names.size
    int64_array = np.int64(np.zeros(rows))
    for r in range(0, rows):
        int64_array[r] = np.int64(int(row_names[r][4:15]))
        
    return int64_array

def ensemble_strings_as_numbers(ens_strs):
    """ for matlab cell array or python list
    Args:
        ens_strs: python list of ensembel gene identifier strings 
    Returns:
        ens_numbers: 64 bit integer array equivalent to the numerical part
    """
    ens_numbers = np.int64(np.zeros(ens_strs.count('\n')))
    ens_str_lines = ens_strs.splitlines()
    row = -1
    for line_string in ens_str_lines:
        row += 1
        ens_numbers[row] = np.int64(float(line_string[4:14]))
        
    return ens_numbers
    
def form_network_laplacian(network):
    """Forms the laplacian matrix.

    Args:
        network: adjancy matrix.

    Returns:
        diagonal_laplacian: the diagonal of the laplacian matrix.
        laplacian: the laplacian matrix.
    """
    laplacian = spar.lil_matrix(network.copy())
    laplacian.setdiag(0)
    laplacian[laplacian != 0] = 1
    diagonal_laplacian = np.array(laplacian.sum(axis=0)) * np.eye(laplacian.shape[0])
    laplacian = laplacian.tocsr()
    diagonal_laplacian = spar.csr_matrix(diagonal_laplacian)
    
    return diagonal_laplacian, laplacian

def project_sample_on_network(raw_spreadsheet, lut, network_size):
    """ project the spreadsheet sample onto the network dirived look up table
    Args:
        raw_spreadsheet: genes x samples (numeric binary) matrix
        lut: size of spreadsheet genes array of indices of gene locations in 
            network or -1 if not present
        network_size: the size of the output spreadsheet genes
        
    Returns:
        spreadsheet: network genes x samples matrix
        columns_removed: any columns that were all zeros after projection
    """
    
    columns = raw_spreadsheet.shape[1]
    lut = lut.T
    spreadsheet = np.zeros((network_size, columns))

    for col_id in range(0, columns):
        nanozeros_in_col = (raw_spreadsheet[:, col_id] != 0)
        index = lut[nanozeros_in_col]
        index = index[(index >= 0) & (index <= network_size)]
        spreadsheet[index, col_id] = 1

    #---------------------------
    # eliminate zero columns
    #---------------------------
    columns_removed = np.arange(0, columns)
    positive_col_set = sum(spreadsheet) > 0
    spreadsheet = spreadsheet[:, positive_col_set]
    columns_removed = columns_removed[np.logical_not(positive_col_set)]
    
    return spreadsheet, columns_removed

def spreadsheet_sample(spreadsheet, percent_sample):
    """Selects a sample, by precentage, from a spread sheet already projected
        on network.

    Args:
        spreadsheet: (network) gene x sample spread sheet.
        percent_sample: percentage of spread sheet to select at random.

    Returns:
        sample_random: A specified precentage sample of the spread sheet.
        patients_permutation: the list the correponds to random sample.
    """
    features_size = np.int_(np.round(spreadsheet.shape[0] * (1-percent_sample)))
    features_permutation = np.random.permutation(spreadsheet.shape[0])
    features_permutation = features_permutation[0:features_size].T
    
    patients_size = np.int_(np.round(spreadsheet.shape[1] * percent_sample))
    patients_permutation = np.random.permutation(spreadsheet.shape[1])
    patients_permutation = patients_permutation[0:patients_size]
    
    sample_random = spreadsheet[:, patients_permutation]
    sample_random[features_permutation[:, None], :] = 0
    
    positive_col_set = sum(sample_random) > 0
    sample_random = sample_random[:, positive_col_set]
    patients_permutation = patients_permutation[positive_col_set]

    return sample_random, patients_permutation

def get_a_sample(spreadsheet, percent_sample, lut, network_size):
    """Selects a sample, by precentage, from a given spread sheet.

    Args:
        spreadsheet: genexsample spread sheet.
        percent_sample: percentage of spread sheet to select at random.
        lut: lookup table.
        network_size: network size

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
    
def get_keg_input(args):
    
    print('get_keg_input is called')
    parser = argparse.ArgumentParser()
    parser.add_argument('-keg_data', '--keg_run_data', type=str)
    parser.add_argument('-target_filename', '--target_file', type=str)
    args = parser.parse_args()
    
    f_name = args.keg_run_data
    print('received filename: {}'.format(f_name))
    network = 1
    spreadsheet = 1
    try:
        data_file = h5py.File(f_name, 'r')
        #network = np.array(data_file["network"])
        #spreadsheet = np.array(data_file["spreadsheet"])
    except:
        print('Failing at except: line 555')
        data_file.close()
        
    Ld = []
    Lk = []
    nbs_par_set = nbs_par_set_dict()
    nbs_par_set["target_filename"] = args.target_file
    data_file.close()
    
    print('get_keg_input Returns')
    
    return network, spreadsheet, Ld, Lk, nbs_par_set
    
def nbs_par_set_dict():
    nbs_par_set = {"number_of_bootstraps":1, "percent_sample":0.8, "k":3,
                    "rwr_alpha":0.7}
    return nbs_par_set


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

def write_connectivity_indicator_matrices(M,I,target_file_name):
    status = 0
    try:
        write_file = h5py.File(target_file_name, 'w')
        M_dataset = write_file.create_dataset('M', (M.shape), dtype=np.float64)
        M_dataset[...] = M
        I_dataset = write_file.create_dataset('I', (I.shape), dtype=np.float64)
        I_dataset[...] = I
        write_file.close()
    except:
        status = -1
    
    return status
    