# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: Sobh
@author: dlanier

"""
import time
import numpy as np
import numpy.linalg as LA
import pandas as pd
import scipy.sparse as spar
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse

    
def nbs_par_set_dict():
    """ set of default parameter names for nbs -- filenames mut be replaced
    
    """
    nbs_par_set = {"number_of_bootstraps":5, "percent_sample":0.8, "k":3,
                    "rwr_alpha":0.7, 'network_file':'network.edge',
                    'spreadsheet_file':'spreadsheet.df',
                    'verbose':1, 'display_clusters':1}
    return nbs_par_set
    
def get_input(args):
    """ read system input arguments and return data from files
    
    åArgs:
        args: aka sys.argv
    
    Returns:
        adj_mat: a symmetric adjacency matrix from the network file input
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        lookup: gene names to location in adj_mat
        rev_lookup: location in adj_mat to gene_names
        par_set_dict: run parameters including the input data filenames
    """
    par_set_dict = get_arg_filenames(args)
    N_df, S_df = read_input_files(par_set_dict)
    adj_mat, spreadsheet = df_to_nw_ss(N_df, S_df)
    adj_mat = normal_matrix(adj_mat)
    if int(par_set_dict['verbose']) != 0:
        echo_input(adj_mat, spreadsheet, par_set_dict)
    
    return adj_mat, spreadsheet, par_set_dict, S_df.columns

def get_arg_filenames(args):
    """ exctract parameters dictionary from command line input args and 
        get the input filenames from that dictionary
        
    Args:
        args: aka sys.argv input to main function
        
    Returns:
        network_file: file name for network .edge
        spreadsheet_file: file name of spreadsheet .df
        par_set: dictionary parameter set
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameters', '--par_data', type=str)
    args = parser.parse_args()    
    f_name = args.par_data
    par_set_df = pd.read_csv(f_name, sep='\t', header=None, index_col=0)
    par_set_dict = dict(par_set_df.to_dict()[1])
    
    return par_set_dict
    
def read_input_files(par_set_dict):
    """ read the input parameters filenames into pandas dataframes
    
    Args:
        network_data_file: network (.edge 4 col) file name
        spreadsheet_data_file: spreadsheet (.df) file name

    Returns:
        N_df: pandas dataframe of network w/o row or col labels
        S_df: pandas dataframe of spreadsheet
    """
    network_file = par_set_dict['network_data']
    spreadsheet_file = par_set_dict['spreadsheet_data']
    N_df = pd.read_table(network_file, sep='\t')
    S_df = pd.read_table(spreadsheet_file, sep='\t', header=0, index_col=0)
    
    return N_df, S_df

def df_to_nw_ss(N_df, S_df):
    """ convert pandas dataframe representations into data set
    
    Args:
        N_df: pandas dataframe of network w/o row or col labels
        S_df: pandas dataframe of spreadsheet

    Returns:
        adj_mat: adjacency matrix (sparse, symmetric, genes x genes)
        spreadsheet: spreadsheet matrix (genes x samples)
        lookup: dictionary of ensembl names to locations in network
        rev_lookup: dictionary locations in network to ensembl names
    """
    from_nodes = N_df.values[:, 0]
    to_nodes = N_df.values[:, 1]
    all_nodes = list(set(from_nodes) | set(to_nodes))
    S_df = S_df.loc[all_nodes].fillna(0)
    lookup = dict(zip(all_nodes, range(len(all_nodes))))
    # rev_lookup = dict(zip(range(len(all_nodes)), all_nodes))
    row_idx = [lookup[i] for i in from_nodes]
    col_idx = [lookup[i] for i in to_nodes]
    N_vals = np.float64(N_df.values[:, 2])

    matrix_length = len(all_nodes)

    adj_mat = spar.csr_matrix((N_vals, (row_idx, col_idx)),
                              shape=(matrix_length, matrix_length))
    adj_mat = adj_mat + adj_mat.transpose()
    spreadsheet = S_df.as_matrix()
    
    return adj_mat, spreadsheet

def normal_matrix(adj_mat):
    """ normalize square matrix for random walk id est.
        normalize s.t. the norm of the whole matrix is near one
    
    Args:
        adj_mat: usually an adjacency matrix
        
    Returns:
        adj_mat: renomralized input s.t. norm(adj_mat) is about 1
    """
    d = sum(adj_mat)
    if spar.issparse(d):
        d = d.todense()
    n_diag = d.size
    d = np.sqrt(1 / d)
    D = np.zeros((n_diag, n_diag))
    for rc in range(0, n_diag):
        D[rc, rc] = d[0, rc]
    D = spar.csr_matrix(D)
    adj_mat = D.dot(adj_mat)
    adj_mat = adj_mat.dot(D.T)
    
    return adj_mat


def form_and_save_h_clusters(adj_mat, spreadsheet, Ld, Lk, nbs_par_set):
    """ main loop for this module computes the components for the consensus matrix
        from the input network and spreadsheet
    Args:
        network: genes x genes symmetric adjacency matrix
        spreadsheet: genes x samples matrix
        Ld, Lk: laplacian matrix components i.e. L = Ld - Lk
        nbs_par_set = {"number_of_bootstraps":1, "percent_sample":0.8, "k":3,
                    "rwr_alpha":0.7}
        number_of_bootstraps: number of iterations of nbs to try
        percent_sample: portion of spreadsheet to use in each iteration
        k: inner dimension of matrx factorization
        alpha: radom walk with restart proportions
    
    Returns:
        connectivity_matrix: samples x samples count of sample relations
        indicator_matrix: samples x samples count of sample trials
    """
    temp_dir = nbs_par_set["temp_dir"]
    number_of_bootstraps = np.int_(nbs_par_set["number_of_bootstraps"])
    k = np.int_(nbs_par_set["k"])
    alpha = np.float64(nbs_par_set["rwr_alpha"])
    
    # ----------------------------------------------
    # Network based clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, number_of_bootstraps):
        sample_random, sample_permutation = spreadsheet_sample(spreadsheet, 
                                    np.float64(nbs_par_set["percent_sample"]))
        sample_smooth, iterations = rwr(sample_random, adj_mat, alpha)
        if int(nbs_par_set['verbose']) != 0:
            print("{} of {}: iterations = {}".format(sample + 1, 
                  number_of_bootstraps, iterations))
        sample_quantile_norm = quantile_norm(sample_smooth)
        H, niter = netnmf(sample_quantile_norm, Lk, Ld, k)
        hname = temp_dir + '/temp_h' + str(sample)
        H.dump(hname)
        pname = temp_dir + '/temp_p' + str(sample)
        sample_permutation.dump(pname)
        
    return
    
def retrieve_h_clusters_and_form_conensus_matrix(nbs_par_set, connectivity_matrix, indicator_matrix):
    """ read the tempfiles and compute the consensus matrix
    
    Args:
        nbs_par_set: parameter set dictionary
        connectivity_matrix: empty connectivity matrix from initialization
        indicator_matrix: empty indicator matrix from initialization
        
    Returns:
        consensus_matrix: connectivity matrices sum / indicator mat sum
    """
    temp_dir = nbs_par_set["temp_dir"]
    number_of_bootstraps = np.int_(nbs_par_set["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        hname = temp_dir + '/temp_h' + str(sample)
        H = np.load(hname)
        pname = temp_dir + '/temp_p' + str(sample)
        sample_permutation = np.load(pname)
        connectivity_matrix = update_connectivity_matrix(H, sample_permutation, connectivity_matrix)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)
        
    consensus_matrix = connectivity_matrix / np.maximum(indicator_matrix, 1)
    
    return consensus_matrix
    
    
def form_network_laplacian(network):
    """ Forms the laplacian matrix.

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
    # spreadsheet = np.float64(spreadsheet_in.copy()) # made no difference
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
    
    return M, labels
    

def echo_input(network, spreadsheet, par_set_dict):
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
    
    for field_n in par_set_dict:
        print('{} : {}'.format(field_n, par_set_dict[field_n]))

    return


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


def write_sample_labels(columns, labels, file_name):
    """ write the .tsv file that attaches a cluster number to the sample name
    Args:
        columns: data identifiers
        labels: cluster numbers
        file_name: write to path name
        
    Returns:
        nothing
    """
    df_tmp = pd.DataFrame(data=labels, index=columns)
    df_tmp.to_csv(file_name, sep='\t', header=None)
    
    return
