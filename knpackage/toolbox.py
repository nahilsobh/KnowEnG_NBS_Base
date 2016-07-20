# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: Sobh
@author: dlanier

"""
# import sessionparameters as ses_par

import argparse
import time
import numpy as np
from numpy import maximum
import numpy.linalg as LA
import pandas as pd
import scipy.sparse as spar
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def nbs_par_set_dict():
    """ Get a set of default parameters for nbs with all possible fielap_dag names.
        Note that the file names and directorys are place holap_dagers only.

    Args: None

    Returns:
        nbs_par_set: a python dictionay of default parameters needed to run the
                     functions in this module.

    nbs_par_set = {"number_of_bootstraps":5,
                   "percent_sample":0.8,
                   "k":3,
                   "rwr_alpha":0.7,
                   'network_data':'network.edge',
                   'spreadsheet_data':'spreadsheet.df',
                   'temp_dir':'/',
                   'consensus_data_df':'co_dat.df',
                   'consensus_data_tsv':'co_dat.tsv',
                   'verbose':1,
                   'display_clusters':1}

    """
    nbs_par_set = {"number_of_bootstraps":5,
                   "percent_sample":0.8,
                   "k":3,
                   "rwr_alpha":0.7,
                   'network_data':'network.edge',
                   'spreadsheet_data':'spreadsheet.df',
                   'temp_dir':'/',
                   'consensus_data_df':'co_dat.df',
                   'consensus_data_tsv':'co_dat.tsv',
                   'verbose':1,
                   'display_clusters':1}

    return nbs_par_set

def get_session_parameters(f_name):
    """ read paramegers file into parameters dictionary
    
    Args:
        f_name: file name
        
    Returns:
        par_set_dict: python dictionary of name - value parameters
    """
    par_set_df = pd.read_csv(f_name, sep='\t', header=None, index_col=0)
    session_parameters = par_set_df.to_dict()[1]
    
    return session_parameters

def get_input(args):
    """ Read system input arguments and return data from the indicated files.

    Args:
        args: aka sys.argv

    Returns:
        adj_mat: a symmetric adjacency matrix from the network file input
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        par_set_dict: run parameters including the input data filenames
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameters', '--par_data', type=str)
    args = parser.parse_args()
    f_name = args.par_data
    par_set_df = pd.read_csv(f_name, sep='\t', header=None, index_col=0)
    par_set_dict = par_set_df.to_dict()[1] # par_set_dict = dict(par_set_df.to_dict()[1])

    network_file = par_set_dict['network_data']
    spreadsheet_file = par_set_dict['spreadsheet_data']
    nw_df = pd.read_table(network_file, sep='\t')
    ss_df = pd.read_table(spreadsheet_file, sep='\t', header=0, index_col=0)

    adj_mat, spreadsheet = df_to_nw_ss(nw_df, ss_df)
    adj_mat = normalized_matrix(adj_mat)
    if int(par_set_dict['verbose']) != 0:
        echo_input(adj_mat, spreadsheet, par_set_dict)
    columns = ss_df.columns
    
    return adj_mat, spreadsheet, par_set_dict, columns


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


def form_and_save_h_clusters(adj_mat, spreadsheet, lap_dag, lap_val, nbs_par_set):
    """ main loop for this module. Computes the components for the consensus
        matrix from the input network and spreadsheet
    Args:
        network: genes x genes symmetric adjacency matrix
        spreadsheet: genes x samples matrix
        lap_dag, lap_val: laplacian matrix components i.e. L = lap_dag - lap_val
        nbs_par_set = {"number_of_bootstraps":1, "percent_sample":0.8, "k":3,
                    "rwr_alpha":0.7}
        number_of_bootstraps: number of iterations of nbs to try
        percent_sample: portion of spreadsheet to use in each iteration
        k: inner dimension of matrx factorization
        alpha: radom walap_val with restart proportions

    Returns:
        connectivity_matrix: samples x samples count of sample relations
        indicator_matrix: samples x samples count of sample trials
    """
    # ----------------------------------------------
    # Network based clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, np.int_(nbs_par_set["number_of_bootstraps"])):
        sample_random, sample_permutation = spreadsheet_sample(spreadsheet,
            np.float64(nbs_par_set["percent_sample"]))
        sample_smooth, iterations = rwr(sample_random, adj_mat,
                                        np.float64(nbs_par_set["rwr_alpha"]))
        if int(nbs_par_set['verbose']) != 0:
            print("{} of {}: iterations = {}".format(sample + 1,
                nbs_par_set["number_of_bootstraps"], iterations))
        sample_quantile_norm = quantile_norm(sample_smooth)
        h_mat = netnmf(sample_quantile_norm, lap_val, lap_dag, np.int_(nbs_par_set["k"]))
        hname = nbs_par_set["temp_dir"] + '/temp_h' + str(sample)
        h_mat.dump(hname)
        pname = nbs_par_set["temp_dir"] + '/temp_p' + str(sample)
        sample_permutation.dump(pname)

    return


def read_h_clusters_to_consensus_matrix(nbs_par_set, connectivity_matrix, indicator_matrix):
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
        h_mat = np.load(hname)
        pname = temp_dir + '/temp_p' + str(sample)
        sample_permutation = np.load(pname)
        connectivity_matrix = update_connectivity_matrix(h_mat,
                                                         sample_permutation, connectivity_matrix)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)

    consensus_matrix = connectivity_matrix / np.maximum(indicator_matrix, 1)

    return consensus_matrix

def normalized_matrix(adj_mat):
    """ normalize square matrix for random walap_val id est.
        normalize s.t. the norm of the whole matrix is near one

    Args:
        adj_mat: usually an adjacency matrix

    Returns:
        adj_mat: renomralized input s.t. norm(adj_mat) is about 1
    """
    row_sm = np.array(adj_mat.sum(axis=0))
    row_sm = 1.0 / row_sm
    row_sm = np.sqrt(row_sm)
    r_c = np.arange(0, adj_mat.shape[0])
    diag_mat = spar.csr_matrix((row_sm[0, :],(r_c, r_c)),shape=(adj_mat.shape))
    adj_mat = diag_mat.dot(adj_mat)
    adj_mat = adj_mat.dot(diag_mat)

    return adj_mat

def form_network_laplacian(adj_mat):
    """Forms the laplacian matrix.

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
    diagonal_laplacian = spar.csr_matrix((rowsum[0, :],(diag_arr, diag_arr)),
                                         shape=(adj_mat.shape))
    laplacian = laplacian.tocsr()
    
    return diagonal_laplacian, laplacian

def spreadsheet_sample(spreadsheet, percent_sample):
    """Selects a sample, by precentage, from a spread sheet already projected
        on adj_mat.

    Args:
        spreadsheet: (adj_mat) gene x sample spread sheet.
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


def rwr(restart, network_sparse, alpha=0.7, max_iteration=100, tol=1.e-4):
    """Performs a random walap_val with restarts.

    Args:
        restart: restart array of any size.
        network_sparse: adjancy matrix stored in sparse format.
        alpha: restart probability. (default = 0.7)
        max_iteration: maximum number of random walap_vals. (default = 100)
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

def update_connectivity_matrix(encode_mat, sample_perm, connectivity_matrix):
    '''Updates the connectivity matrix

    Args:
        h_matrix: nonnegative right factor (H)
        P: sample permutaion of h_matrix
        M: connectivity matrix

    Returns:
        M: modified connectivity matrix
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
        P: sample permutaion of h_matrix
        I: indicator matrix

    Returns:
        I: modified indicator matrix
    '''
    indicator_matrix[sample_perm[:, None], sample_perm] += 1

    return indicator_matrix

def get_labels(consensus_matrix, k=3):
    cluster_handle = KMeans(k, random_state=10)
    labels = cluster_handle.fit_predict(consensus_matrix)
    
    return labels

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


def echo_input(network, spreadsheet, par_set_dict):
    '''Prints User's spread sheet and network data Dimensions and sizes

    Args:
         network: full gene-gene network
         spreadsheet: user's genes x samples data
         par_set_dict: run parameters dictionary

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

    for fielap_dag_n in par_set_dict:
        print('{} : {}'.format(fielap_dag_n, par_set_dict[fielap_dag_n]))

    return


def initialization(spreadsheet):
    '''Initializes connectivity and indicator matrices.

    Args:
         network: full gene-gene network
         spreadsheet: user's data

    Returns:
        network_size :
        M :
        I :
    '''
    sp_size = spreadsheet.shape[1]
    connectivity_matrix = np.zeros((sp_size, sp_size))
    indicator_matrix = np.zeros((sp_size, sp_size))

    return  connectivity_matrix, indicator_matrix


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


def write_output(consensus_matrix, columns, labels, file_name):
    """ save the consensus matrix as a dataframe with column names and row
        cluster number labels

    Args:
        columns: data identifiers for column names
        labels: cluster numbers for row names
        file_name: write to path name

    Returns:
        nothing - just writes the file
    """
    out_df = pd.DataFrame(data=consensus_matrix, columns=columns, index=labels)
    out_df.to_csv(file_name, sep='\t')

    return


def write_sample_labels(columns, labels, file_name):
    """ two column file that attaches a cluster number to the sample name
    Args:
        columns: data identifiers
        labels: cluster numbers
        file_name: write to path name

    Returns:
        nothing - just writes the file
    """
    df_tmp = pd.DataFrame(data=labels, index=columns)
    df_tmp.to_csv(file_name, sep='\t', header=None)

    return
