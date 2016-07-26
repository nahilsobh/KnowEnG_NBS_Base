# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: Sobh
@author: dlanier

"""
import argparse

import time
import numpy as np
from numpy import maximum
import numpy.linalg as LA
import pandas as pd
import scipy.sparse as spar
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ----------------------------------------------
#   Begin: Read and Prepare Input.
# ----------------------------------------------


def get_input(args):
    """ Read system input arguments (argv) to get the run directory name.

    Args:
        args: sys.argv (command line input to main())

    Returns:
        run_directory: run directory name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    args = parser.parse_args()
    run_directory = args.run_directory 

    return run_directory


def get_run_parameters(run_file, run_directory):
    """ Read parameters file into session parameters dictionary.

    Args:
        run_file: run parameters file name.
        run_directory: run directory name.

    Returns:
        run_parameters: python dictionary of name - value parameters.
    """
    f_name = run_directory+'/'+run_file
    par_set_df = pd.read_csv(f_name, sep='\t', header=None, index_col=0)
    
    run_parameters = par_set_df.to_dict()[1]
    run_parameters["run_directory"]=run_directory

    return run_parameters


def get_netnmf_input(run_parameters):
    """ get input arguments for network based non-negative matrix factroization.

    Args:
        run_parameters: parameter set structure of pytyon dictionary type

    Returns:
        adj_mat: adjacency matrix
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        sample_names: column names of spreadsheet data
        lap_diag: diagonal component of laplacian matrix
        lap_pos: positional component of laplacian matrix
    """
    network_file = run_parameters['network_file_name']
    spreadsheet_file = run_parameters['samples_file_name']
    nw_df = pd.read_table(network_file, sep='\t')
    ss_df = pd.read_table(spreadsheet_file, sep='\t', header=0, index_col=0)
    adj_mat, spreadsheet = df_to_nw_ss(nw_df, ss_df)

    adj_mat = normalized_matrix(adj_mat)

    lap_diag, lap_pos = form_network_laplacian(adj_mat)
    sample_names = ss_df.columns
    if int(run_parameters['verbose']) != 0:
        echo_input(adj_mat, spreadsheet, run_parameters)

    return adj_mat, spreadsheet, sample_names, lap_diag, lap_pos


def df_to_nw_ss(nw_df, ss_df):
    """ convert pandas dataframe representations into data set

    Args:
        nw_df: pandas dataframe of network w/o row or col labels
        ss_df: pandas dataframe of spreadsheet

    Returns:
        adj_mat: adjacency matrix (sparse, symmetric, genes x genes)
        spreadsheet: spreadsheet matrix (genes x samples)
        lookup: dictionary of ensembl names to locations in network
        rev_lookup: dictionary locations in network to ensembl names
    """
    from_nodes = nw_df.values[:, 0]
    to_nodes = nw_df.values[:, 1]
    all_nodes = list(set(from_nodes) | set(to_nodes))
    ss_df = ss_df.loc[all_nodes].fillna(0)
    lookup = dict(zip(all_nodes, range(len(all_nodes))))
    # rev_lookup = dict(zip(range(len(all_nodes)), all_nodes))
    row_idx = [lookup[i] for i in from_nodes]
    col_idx = [lookup[i] for i in to_nodes]
    n_vals = np.float64(nw_df.values[:, 2])

    matrix_length = len(all_nodes)

    adj_mat = spar.csr_matrix((n_vals, (row_idx, col_idx)),
                              shape=(matrix_length, matrix_length))
    adj_mat = adj_mat + adj_mat.T
    spreadsheet = ss_df.as_matrix()

    return adj_mat, spreadsheet


def get_nmf_input(run_parameters):
    """ get input arguments for non-negative matrix factroization.

    Args:
        run_parameters: parameter set structure of pytyon dictionary type

    Returns:
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        sample_names: column names of spreadsheet data
    """
    spreadsheet_file = run_parameters['samples_file_name']
    ss_df = pd.read_table(spreadsheet_file, sep='\t', header=0, index_col=0)
    spreadsheet = ss_df.as_matrix()

    if int(run_parameters['verbose']) != 0:
        echo_input(np.zeros((1, 1)), spreadsheet, run_parameters)
    sample_names = ss_df.columns

    return spreadsheet, sample_names


# ----------------------------------------------
#   End: Read and Prepare Input.
#
#
#   Begin: Function Sequence Wrappers.
# ----------------------------------------------


def run_cc_net_nmf(run_parameters):
    """ Wrapper for call sequence that performs network based stratification
        with consensus clustering.

    Args:
        run_parameters: parameter set dictionary

    Returns: (no variables)
        writes consensus matrix as data frame with column names and labels.
        writes table of sample names with cluster assignments.
    """
    adj_mat, spreadsheet, sample_names, lap_diag, lap_pos = get_netnmf_input(run_parameters)

    form_and_save_h_clusters(adj_mat, spreadsheet, lap_diag, lap_pos, run_parameters)

    connectivity_matrix, indicator_matrix = initialization(spreadsheet)
    consensus_matrix = read_h_clusters_to_consensus_matrix(
        run_parameters, connectivity_matrix, indicator_matrix)
    labels = get_labels(consensus_matrix, int(run_parameters['k']))

    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    write_sample_labels(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = reorder_matrix(consensus_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return


def run_net_nmf(run_parameters):
    """ Wrapper for call sequence that performs network based stratification

    Args:
        run_parameters: parameter set dictionary

    Returns: (no variables)
        writes table of sample names with cluster assignments.
    """
    adj_mat, spreadsheet, sample_names, lap_diag, lap_pos = get_netnmf_input(run_parameters)
    sample_smooth, iterations = rwr(spreadsheet, adj_mat,
                                    np.float64(run_parameters["restart_probability"]))
    sample_quantile_norm = quantile_norm(sample_smooth)
    h_mat = netnmf(sample_quantile_norm, lap_pos, lap_diag, np.int_(run_parameters["k"]))

    sp_size = spreadsheet.shape[1]
    connectivity_matrix = np.zeros((sp_size, sp_size))
    sample_perm = np.arange(0, sp_size)
    connectivity_matrix = update_connectivity_matrix(h_mat, sample_perm, connectivity_matrix)
    labels = get_labels(connectivity_matrix, np.int_(run_parameters["k"]))

    write_sample_labels(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = reorder_matrix(connectivity_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return


def run_cc_nmf(run_parameters):
    """ Wrapper for call sequence that performs non-negative matrix factorization
        with consensus clustering.

    Args:
        run_parameters: parameter set dictionary

    Returns: (no variables)
        writes consensus matrix as data frame with column names and labels.
        writes table of sample names with cluster assignments.
    """
    spreadsheet, sample_names = get_nmf_input(run_parameters)

    nmf_form_save_h_clusters(spreadsheet, run_parameters)

    connectivity_matrix, indicator_matrix = initialization(spreadsheet)
    consensus_matrix = read_h_clusters_to_consensus_matrix(
        run_parameters, connectivity_matrix, indicator_matrix)
    labels = get_labels(consensus_matrix, int(run_parameters['k']))

    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    write_sample_labels(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = reorder_matrix(consensus_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return


def run_nmf(run_parameters):
    """ Wrapper for call sequence that performs non-negative matrix factorization

    Args:
        run_parameters: parameter set dictionary

    Returns: (no variables)
        writes table of sample names with cluster assignments.
    """
    spreadsheet, sample_names = get_nmf_input(run_parameters)

    h_mat = nmf(spreadsheet, np.int_(run_parameters["k"]))
    # labels = np.argmax(h_mat, 0)

    sp_size = spreadsheet.shape[1]
    connectivity_matrix = np.zeros((sp_size, sp_size))
    sample_perm = np.arange(0, sp_size)
    connectivity_matrix = update_connectivity_matrix(h_mat, sample_perm, connectivity_matrix)
    labels = get_labels(connectivity_matrix, np.int_(run_parameters["k"]))

    write_sample_labels(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = reorder_matrix(connectivity_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return



def form_and_save_h_clusters(adj_mat, spreadsheet, lap_dag, lap_val, run_parameters):
    """ Computes the components for the consensus matrix from the input network and spreadsheet
        for network based stratification

    Args:
        adj_mat: genes x genes symmetric adjacency matrix
        spreadsheet: genes x samples matrix
        lap_dag, lap_val: laplacian matrix components i.e. L = lap_dag - lap_val
        run_parameters: dictionay of run-time parameters

    Returns: (nothing)
        writes anonymous bootstrap tmp files for h-matrix and its permutation
    """
    # ----------------------------------------------
    # Network based clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, np.int_(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = spreadsheet_sample(spreadsheet,
                                            np.float64(run_parameters["percent_sample"]))
        sample_smooth, iterations = rwr(sample_random, adj_mat,
                                        np.float64(run_parameters["restart_probability"]))
        if int(run_parameters['verbose']) != 0:
            print("{} of {}: iterations = {}".format(
                sample + 1,
                run_parameters["number_of_bootstraps"],
                iterations))

        sample_quantile_norm = quantile_norm(sample_smooth)
        h_mat = netnmf(sample_quantile_norm, lap_val, lap_dag, np.int_(run_parameters["k"]))
        hname = run_parameters["tmp_directory"] + '/temp_h' + str(sample)
        h_mat.dump(hname)
        pname = run_parameters["tmp_directory"] + '/temp_p' + str(sample)
        sample_permutation.dump(pname)

    return


def nmf_form_save_h_clusters(spreadsheet, run_parameters):
    """ Computes the components for the non-negative matric factorization
        consensus matrix from the input spreadsheet.

    Args:
        spreadsheet: genes x samples matrix
        run_parameters: dictionay of run-time parameters

    Returns: (nothing)
        writes anonymous bootstrap tmp files for h-matrix and its permutation
    """
    # ----------------------------------------------
    # nmf clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, np.int_(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = spreadsheet_sample(spreadsheet,
                                                np.float64(run_parameters["percent_sample"]))
        h_mat = nmf(sample_random, np.int_(run_parameters["k"]))
        hname = run_parameters["tmp_directory"] + '/temp_h' + str(sample)
        h_mat.dump(hname)
        pname = run_parameters["tmp_directory"] + '/temp_p' + str(sample)
        sample_permutation.dump(pname)

        if int(run_parameters['verbose']) != 0:
            print('nmf {} of {}'.format(
                sample + 1, run_parameters["number_of_bootstraps"]))

    return


def read_h_clusters_to_consensus_matrix(run_parameters, connectivity_matrix, indicator_matrix):
    """ read anonymous bootstrap tmp files compute the consensus matrix for
        whichever method wrote them.

    Args:
        run_parameters: parameter set dictionary
        connectivity_matrix: connectivity matrix from initialization or previous
        indicator_matrix: indicator matrix from initialization or previous

    Returns:
        consensus_matrix: sum of connectivity matrices / indicator matrices sum
    """
    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = np.int_(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        hname = tmp_dir + '/temp_h' + str(sample)
        h_mat = np.load(hname)
        pname = tmp_dir + '/temp_p' + str(sample)
        sample_permutation = np.load(pname)
        connectivity_matrix = update_connectivity_matrix(h_mat,
                                                         sample_permutation, connectivity_matrix)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)

    consensus_matrix = connectivity_matrix / np.maximum(indicator_matrix, 1)

    return consensus_matrix


# ----------------------------------------------
#   End: Function Sequence Wrappers.
#
#
#   Begin: Core Functions.
# ----------------------------------------------


def normalized_matrix(adj_mat):
    """ normalize symmetrix matrix s.t. the norm of the whole matrix is near one

    Args:
        adj_mat: symmetric adjacency matrix

    Returns:
        adj_mat: input matrix - renomralized
    """
    row_sm = np.array(adj_mat.sum(axis=0))
    row_sm = 1.0 / row_sm
    row_sm = np.sqrt(row_sm)
    r_c = np.arange(0, adj_mat.shape[0])
    diag_mat = spar.csr_matrix((row_sm[0, :], (r_c, r_c)), shape=(adj_mat.shape))
    adj_mat = diag_mat.dot(adj_mat)
    adj_mat = adj_mat.dot(diag_mat)

    return adj_mat

def form_network_laplacian(adj_mat):
    """Forms the laplacian matrix components for use in network based stratification.

    Args:
        adj_mat: adjancy matrix.

    Returns:
        diagonal_laplacian: the diagonal of the laplacian matrix.
        laplacian: the laplacian matrix.
    """
    laplacian = spar.lil_matrix(adj_mat.copy())
    laplacian.setdiag(0)
    laplacian[laplacian != 0] = 1
    diag_length = laplacian.shape[0]
    rowsum = np.array(laplacian.sum(axis=0))
    diag_arr = np.arange(0, diag_length)
    diagonal_laplacian = spar.csr_matrix((rowsum[0, :], (diag_arr, diag_arr)),
                                         shape=(adj_mat.shape))
    laplacian = laplacian.tocsr()

    return diagonal_laplacian, laplacian

def spreadsheet_sample(spreadsheet, percent_sample):
    """ Select a (fraction x fraction)sample, from a spreadsheet

    Args:
        spreadsheet: (adj_mat) gene x sample spread sheet.
        percent_sample: selection "percentage" - [0 : 1]

    Returns:
        sample_random: A specified precentage sample of the spread sheet.
        sample_permutation: the array that correponds to random sample.
    """
    features_size = np.int_(np.round(spreadsheet.shape[0] * (1-percent_sample)))
    features_permutation = np.random.permutation(spreadsheet.shape[0])
    features_permutation = features_permutation[0:features_size].T

    patients_size = np.int_(np.round(spreadsheet.shape[1] * percent_sample))
    sample_permutation = np.random.permutation(spreadsheet.shape[1])
    sample_permutation = sample_permutation[0:patients_size]

    sample_random = spreadsheet[:, sample_permutation]
    sample_random[features_permutation[:, None], :] = 0

    positive_col_set = sum(sample_random) > 0
    sample_random = sample_random[:, positive_col_set]
    sample_permutation = sample_permutation[positive_col_set]

    return sample_random, sample_permutation


def rwr(restart, network_sparse, alpha=0.7, max_iteration=100, tol=1.e-4):
    """ Simulates a random walk with restarts.

    Args:
        restart: restart array of any size.
        network_sparse: adjancy matrix stored in sparse format.
        alpha: restart probability. (default = 0.7)
        max_iteration: maximum number of random walap_vals. (default = 100)
        tol: convergence tolerance. (default = 1.e-4)
        report_frequency: frequency of convergance checks. (default = 5)

    Returns:
        smooth_1: smoothed restart data.
        step: number of iterations used
    """
    smooth_0 = restart
    smooth_r = (1. - alpha) * restart
    for step in range(0, max_iteration):
        smooth_1 = alpha * network_sparse.dot(smooth_0) + smooth_r
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
    index = np.argsort(sample, axis=0)           # each column determine rank
    sample_sorted_by_rows = np.sort(sample, axis=0)   # Sort each column
    mean_per_row = sample_sorted_by_rows.mean(1) # each row determine its mean
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
    h_pos = h_matrix > 0
    h_matrix[~h_pos] = 0
    col_log_arr = sum(h_pos == 0) > 0
    col_list = colix[col_log_arr]
    for cluster in range(0, number_of_clusters):
        if col_list.size > 0:
            w_ette = wtx[:, col_list]
            m_rows = w_ette.shape[0]
            n_cols = w_ette.shape[1]
            mcode_uniq_col_ix = np.arange(0, n_cols)
            h_ette = np.zeros((m_rows, n_cols))
            h_pos_ette = h_pos[:, col_list]
            mcoding = np.dot(2**(np.arange(0, m_rows)), np.int_(h_pos_ette))
            mcode_uniq = np.unique(mcoding)
            for u_n in mcode_uniq:
                ixidx = mcoding == u_n
                c_pat = mcode_uniq_col_ix[ixidx]
                if c_pat.size > 0:
                    r_pat = rowix[h_pos_ette[:, c_pat[0]]]
                    atmp = wtw[r_pat[:, None], r_pat]
                    btmp = w_ette[r_pat[:, None], c_pat]
                    atmptatmp = np.dot(atmp.T, atmp)
                    atmptatmp = LA.pinv(atmptatmp)
                    atmptbtmp = np.dot(atmp.T, btmp)
                    h_ette[r_pat[:, None], c_pat] = np.dot(atmptatmp, atmptbtmp)
                    h_matrix[:, col_list] = h_ette
            h_pos = h_matrix > 0
            h_matrix[~h_pos] = 0
            col_log_arr = sum(h_pos == 0) > 0
            col_list = colix[col_log_arr]
        else:
            break
    return h_matrix


def netnmf(x_matrix, lap_val, lap_dag, k=3, lmbda=1400, it_max=10000, h_clust_eq_limit=200,
           obj_fcn_chk_freq=50):
    """Performs network based nonnegative matrix factorization that
    minimizes( ||X-WH|| + lambda.tr(W'.L.W)

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W.H
        lap_val: the laplacian matrix
        lap_dag: the diagonal of the laplacian matrix
        k: number of clusters
        lmbda: penalty numnber (default = 100)
        it_max: maximim objective function iterations (default = 10000)
        h_clust_eq_limit: h_matrix no change objective (default = 200)
        obj_fcn_chk_freq: objective function check interval (default = 50)

    Returns:
        h_matrix: nonnegative right factor (H)
        itr: number of iterations completed
    """
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, it_max):
        if np.mod(itr, obj_fcn_chk_freq) == 0:
            h_clusters = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(h_clust_eq != h_clusters) == 0):
                h_eq_count = h_eq_count + obj_fcn_chk_freq
            else:
                h_eq_count = 0
            h_clust_eq = h_clusters
            if h_eq_count >= h_clust_eq_limit:
                break
        numerator = maximum(np.dot(x_matrix, h_matrix.T) + lmbda * lap_val.dot(w_matrix), epsilon)
        denomerator = maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T))
                              + lmbda * lap_dag.dot(w_matrix), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = get_h(w_matrix, x_matrix)

    return h_matrix


def nmf(x_matrix, k=3, it_max=10000, h_clust_eq_limit=200, obj_fcn_chk_freq=50):
    """Performs nonnegative matrix factorization that minimizes ||X-WH||

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W.H
        k: number of clusters
        it_max: maximim objective function iterations (default = 10000)
        h_clust_eq_limit: h_matrix no change objective (default = 200)
        obj_fcn_chk_freq: objective function check interval (default = 50)

    Returns:
        h_matrix: nonnegative right factor (H)
    """
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, it_max):
        if np.mod(itr, obj_fcn_chk_freq) == 0:
            h_clusters = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(h_clust_eq != h_clusters) == 0):
                h_eq_count = h_eq_count + obj_fcn_chk_freq
            else:
                h_eq_count = 0
            h_clust_eq = h_clusters
            if h_eq_count >= h_clust_eq_limit:
                break
        numerator = maximum(np.dot(x_matrix, h_matrix.T), epsilon)
        denomerator = maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T)), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = get_h(w_matrix, x_matrix)
        #h_matrix = h_matrix * np.dot(w_matrix.T, x_matrix) / np.dot(
                                        #np.dot(w_matrix.T, w_matrix), h_matrix)

    return h_matrix


def initialization(spreadsheet):
    ''' Initializes connectivity and indicator matrices.

    Args:
         spreadsheet: user's data

    Returns:
        connectivity_matrix: samples x samples matrix of zeros
        indicator_matrix: samples x samples matrix of zeros
    '''
    sp_size = spreadsheet.shape[1]
    connectivity_matrix = np.zeros((sp_size, sp_size))
    indicator_matrix = np.zeros((sp_size, sp_size))

    return  connectivity_matrix, indicator_matrix


def update_connectivity_matrix(encode_mat, sample_perm, connectivity_matrix):
    '''Updates the connectivity matrix

    Args:
        encode_mat: nonnegative right factor (H)
        sample_perm: sample permutaion of h_matrix
        connectivity_matrix: connectivity matrix

    Returns:
        connectivity_matrix: modified connectivity matrix
    '''
    num_clusters = encode_mat.shape[0]
    cluster_id = np.argmax(encode_mat, 0)
    for cluster in range(0, num_clusters):
        slice_id = sample_perm[cluster_id == cluster]
        connectivity_matrix[slice_id[:, None], slice_id] += 1
    return connectivity_matrix


def update_indicator_matrix(sample_perm, indicator_matrix):
    '''Updates the indicator matrix.

    Args:
        sample_perm: sample permutaion of h_matrix
        indicator_matrix: indicator matrix

    Returns:
        indicator_matrix: modified indicator matrix
    '''
    indicator_matrix[sample_perm[:, None], sample_perm] += 1

    return indicator_matrix


def get_labels(consensus_matrix, k=3):
    """ determine cluster assignments for consensus matrix

    Args:
        consensus_matrix: connectivity / indicator matrices
        k: clusters estimate

    Returns:
        lablels: ordered cluster assignments for consensus_matrix
    """
    cluster_handle = KMeans(k, random_state=10)
    labels = cluster_handle.fit_predict(consensus_matrix)

    return labels


# ----------------------------------------------
#   End: Core Functions.
#
#
#   Begin: Runtime Utility.
# ----------------------------------------------


def reorder_matrix(consensus_matrix, k=3):
    '''Performs K-means and use its labels to reorder the consensus matrix

    Args:
        consensus_matrix: unordered consensus
        k: number of clusters

    Returns:
        M: ordered consensus
    '''
    cc_cm = consensus_matrix.copy()
    labels = get_labels(consensus_matrix, k)
    sorted_labels = np.argsort(labels)
    cc_cm = cc_cm[sorted_labels[:, None], sorted_labels]

    return cc_cm


def echo_input(network, spreadsheet, run_parameters):
    '''Prints User's spread sheet and network data Dimensions and sizes

    Args:
         network: full gene-gene network
         spreadsheet: user's genes x samples data
         run_parameters: run parameters dictionary

    Returns:
        nothing - just displays the input data to the command line
    '''
    net_rows = network.shape[0]
    net_cols = network.shape[1]
    usr_rows = spreadsheet.shape[0]
    usr_cols = spreadsheet.shape[1]
    date_frm = "Local: %a, %d %b %Y %H:%M:%S"

    print('Data Loaded:\t{}'.format(time.strftime(date_frm, time.localtime())))
    print('adjacency    matrix {} x {}'.format(net_rows, net_cols))
    print('spread sheet matrix {} x {}'.format(usr_rows, usr_cols))

    for fielap_dag_n in run_parameters:
        print('{} : {}'.format(fielap_dag_n, run_parameters[fielap_dag_n]))

    return


def display_clusters(consensus_matrix):
    '''Displays the consensus matrix.

    Args:
         M: consenus matrix.

    Returns:
        nothing - just displays the matrix as a heat map
    '''
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    grid = consensus_matrix
    fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    for ax_n, interp_method in zip(axes.flat, methods):
        ax_n.imshow(grid, interpolation=interp_method)
        ax_n.set_title(interp_method)
    plt.show()

    return


def write_consensus_matrix(consensus_matrix, columns, labels, run_parameters):
    """ save the consensus matrix as a dataframe with column names and row
        cluster number labels

    Args:
        consensus_matrix: labels x columns numerical matrix
        columns: data identifiers for column names
        labels: cluster numbers for row names
        file_name: write to path name

    Returns:
        nothing - just writes the file
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = now_name('consensus_data', 'df')
    else:
        file_name = 'consensus_data.df'
    out_df = pd.DataFrame(data=consensus_matrix, columns=columns, index=labels)
    out_df.to_csv(file_name, sep='\t')

    return


def write_sample_labels(columns, labels, run_parameters):
    """ two column file that attaches a cluster number to the sample name

    Args:
        columns: data identifiers (unique)
        labels: cluster number assignments (not unique)
        file_name: write path and file name

    Returns:
        nothing - writes the file
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = now_name('labels_data', 'tsv')
    else:
        file_name = 'labels_data.tsv'
    df_tmp = pd.DataFrame(data=labels, index=columns)
    df_tmp.to_csv(file_name, sep='\t', header=None)

    return


def now_name(name_base, name_extension, delta_time=1e6):
    """ insert a time stamp into the filename with estension

    Args:
        name_base: file name first part - may include directory path
        name_extension: file extension without a period
        delta_time: 1e6 equates to microsecond step size

    Returns:
        time_stamped_file_name: concatenation of the inputs with time-stamp
    """
    nstr = np.str_(np.int_(time.time() * delta_time))
    time_stamped_file_name = name_base + '_' + nstr + '.' + name_extension

    return time_stamped_file_name


def run_parameters_dict():
    """ Dictionary of parameter field names with default values.

    Args: None

    Returns:
        run_parameters: a python dictionay of default parameters needed to run the
            functions in this module. Write to text file:
        par_dataframe = pd.DataFrame.from_dict(run_parameters, orient='index')
        par_dataframe.to_csv(file_name, sep='\t')
    """
    run_parameters = {
        "method":"cc_net_nmf",
        "k":4,
        "number_of_bootstraps":5,
        "percent_sample":0.8,
        "restart_probability":0.7,
        "number_of_iterations":100,
        "tolerance":1e-4,
        "network_threshold":1.0,
        "network_etype":"None",
        "network_taxon":"None",
        "property_etype":"None",
        "property_taxon":"None",        
        "network_file_name":"network_file_name",
        "samples_file_name":"samples_file_name",
        "tmp_directory":"tmp",
        "run_directory":"run_directory",
        "use_now_name":1,
        "verbose":1,
        "display_clusters":1}

    return run_parameters


def generate_run_file(file_name='run_file'):
    """ Write a defaut parameters set to a text file for editing

    Args:
        file_name: file name with extension (will be written as plain text).

    Returns:
        no variables, writes a default parameters dictionary file.

    """
    par_dataframe = pd.DataFrame.from_dict(run_parameters_dict(), orient='index')
    par_dataframe.to_csv(file_name, sep='\t')

    return
