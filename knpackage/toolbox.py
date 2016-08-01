# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: The Gene Sets Characterization dev team

"""
import os
import time
import argparse
import numpy as np
import numpy.linalg as LA
import pandas as pd
import scipy.sparse as spar
import matplotlib.pyplot as plt

from numpy import maximum
from sklearn.cluster import KMeans

def get_run_directory(args):
    """ Read system input arguments (argv) to get the run directory name

    Args:
        args: sys.argv, command line input; python main -run_directory dir_name

    Returns:
        run_directory: directory where run_file is expected
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    args = parser.parse_args()
    run_directory = args.run_directory

    return run_directory

def get_run_parameters(run_directory):
    """ Read system input arguments run directory name and run_file into a dictionary.

    Args:
        run_directory: directory where run_file is expected

    Returns:
        run_parameters: python dictionary of name - value parameters.
    """
    run_file_name = os.path.join(run_directory, "run_file")
    par_set_df = pd.read_csv(run_file_name, sep='\t', header=None, index_col=0)
    run_parameters = par_set_df.to_dict()[1]
    run_parameters["run_directory"] = run_directory

    return run_parameters

def get_spreadsheet(run_parameters):
    """ get the spreadsheet file name from the run_parameters dictionary and
        read the file into a pandas dataframe.

    Args:
        run_parameters: python dictionary with 'samples_file_name' key.

    Returns:
        spreadsheet_df: the spreadsheet dataframe.
    """
    spreadsheet_df = pd.read_csv(
        run_parameters['samples_file_name'], sep='\t', header=0, index_col=0)

    return spreadsheet_df

def get_network(run_parameters):
    """ get the network file name from the run_parameters dictionary and
        read the file into a pandas dataframe.

    Args:
        run_parameters: python dictionary with 'samples_file_name' key.

    Returns:
        network_df: the network dataframe.
    """
    network_df = pd.read_csv(run_parameters['network_file_name'], sep='\t',
                             header=None, usecols=[0, 1, 2])

    return network_df

def get_network_node_names(network_df):
    """ get the set (list) of all genes in the network dataframe

    Args:
        network_df: pandas dataframe of network input file

    Returns:
        gene_do_list: list of network genes
    """
    from_nodes = network_df.values[:, 0]
    to_nodes = network_df.values[:, 1]
    gene_do_list = list(set(from_nodes) | set(to_nodes))

    return gene_do_list

def get_spreadsheet_gene_names(spreadsheet_df):
    """ get the set (list) of all genes in the spreadsheet dataframe

    Args:
        spreadsheet_df: dataframe of spreadsheet input file

    Returns:
        spreadsheet_genes: list of spreadsheet genes
    """
    spreadsheet_genes = spreadsheet_df.index.values

    return spreadsheet_genes

def write_spreadsheet_droplist(spreadsheet_df, gene_do_list, run_parameters):
    """ write the list of genes that are in the input spreadsheed and not in the
        gene_do_list to the droplist_Fisher.txt in run_parameters tmp_directory
    Args:
        spreadsheet_df: the full spreadsheet data frame before dropping
        gene_do_list: the genes that will be used in calculation
        run_parameters: dictionary of parameters
    """
    tmp_dir = run_parameters['tmp_directory']
    droplist = spreadsheet_df.loc[~spreadsheet_df.index.isin(gene_do_list)]
    file_path = os.path.join(tmp_dir, "droplist_Fisher.txt")
    droplist.to_csv(file_path, sep='\t')

    return

def update_spreadsheet_fill(spreadsheet_df, gene_do_list):
    """ resize and reorder spreadsheet dataframe to only the genes in the network

    Args:
        spreadsheet_df: pandas dataframe of spreadsheet
        gene_do_list: python list of all genes in network

    Returns:
        spreadsheet_df: pandas dataframe of spreadsheet with only network genes
    """
    spreadsheet_df = spreadsheet_df.loc[gene_do_list].fillna(0)

    return spreadsheet_df

def update_spreadsheet_drop(spreadsheet_df, gene_do_list):
    """ resize and reorder spreadsheet dataframe to only the genes in the network

    Args:
        spreadsheet_df: pandas dataframe of spreadsheet
        gene_do_list: python list of all genes in network

    Returns:
        spreadsheet_df: pandas dataframe of spreadsheet with only network genes
    """
    spreadsheet_df = spreadsheet_df.loc[spreadsheet_df.index.isin(gene_do_list)]

    return spreadsheet_df

def create_genes_lookup_table(gene_do_list):
    """ create a python dictionary to look up gene locations from gene names

    Args:
        gene_do_list: python list of gene names

    Returns:
        genes_lookup_table: python dictionary of gene names to integer locations
    """
    genes_lookup_table = dict(zip(gene_do_list, range(len(gene_do_list))))

    return genes_lookup_table

def symmetrize_df(network_df):
    """ create matrix symmetry by appending network data frame to itself while
        swapping col 0 and col 1 in the bottom half

    Args:
        network_df: 3 or 4 column pandas data frame

    Returns:
        symm_network_df:
    """
    n_df = network_df.copy()
    n_df.loc[n_df.index[:], n_df.columns[0]] = network_df.loc[n_df.index[:], n_df.columns[1]]
    n_df.loc[n_df.index[:], n_df.columns[1]] = network_df.loc[n_df.index[:], n_df.columns[0]]
    symm_network_df = pd.concat([network_df, n_df])
    symm_network_df.index = np.arange(0, symm_network_df.shape[0])

    return symm_network_df

def map_network_names(network_df, genes_lookup_table):
    """ replace the node names with numbers for input to sparse matrix

    Args:
        network_df:
        genes_lookup_table:

    Returns:
        network_df: the same dataframe with integer
    """
    from_nodes = network_df.values[:, 0]
    to_nodes = network_df.values[:, 1]

    row_idx = np.int_(np.array([genes_lookup_table[i] for i in from_nodes]))
    col_idx = np.int_(np.array([genes_lookup_table[i] for i in to_nodes]))
    tmp_dict = {'col_0':row_idx, 'col_1':col_idx, 'col_2':network_df.values[:, 2]}
    network_numeric_df = pd.DataFrame(tmp_dict)

    return network_numeric_df

def convert_df_to_sparse(network_df, matrix_length):
    """ convert network dataframe with numerical columns to scipy.sparse matrix

    Args:
        network_df: padas dataframe with numerical columns (data, row_ix, col_ix)
        matrix_length: size of square "network_sparse" matrix output

    Returns:
        network_sparse: scipy.sparse.csr_matrix
    """
    nwm = network_df.as_matrix()
    network_sparse = spar.csr_matrix((np.float_(nwm[:, 2]),
                                      (np.int_(nwm[:, 0]), np.int_(nwm[:, 1]))),
                                      shape=(matrix_length, matrix_length))

    return network_sparse

def get_net_nmf_input(run_parameters):
    """ get input arguments for network based non-negative matrix factroization
        using file names specified in the run_parameters

    Args:
        run_parameters: parameter set structure of pytyon dictionary type

    Returns:
        adj_mat: adjacency matrix
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        sample_names: column names of spreadsheet data
        lap_diag: diagonal component of laplacian matrix
        lap_pos: positional component of laplacian matrix
    """
    network_df = get_network(run_parameters)
    spreadsheet_df = get_spreadsheet(run_parameters)

    gene_do_list = get_network_node_names(network_df)
    genes_lookup_table = create_genes_lookup_table(gene_do_list)
    network_df = map_network_names(network_df, genes_lookup_table)
    network_df = symmetrize_df(network_df)
    adj_mat = convert_df_to_sparse(network_df, len(gene_do_list))

    adj_mat = normalized_matrix(adj_mat)
    lap_diag, lap_pos = form_network_laplacian(adj_mat)

    spreadsheet_df = update_spreadsheet_fill(spreadsheet_df, gene_do_list)
    spreadsheet = spreadsheet_df.as_matrix()
    sample_names = spreadsheet_df.columns

    if int(run_parameters['verbose']) != 0:
        echo_input(adj_mat, spreadsheet, run_parameters)

    return adj_mat, spreadsheet, sample_names, lap_diag, lap_pos

def get_nmf_input(run_parameters):
    """ get input arguments for non-negative matrix factroization.

    Args:
        run_parameters: parameter set structure of pytyon dictionary type

    Returns:
        spreadsheet: genes x samples input data matrix shaped to adj_mat
        sample_names: column names of spreadsheet data
    """
    ss_df = get_spreadsheet(run_parameters)
    spreadsheet = ss_df.as_matrix()
    spreadsheet = get_quantile_norm(spreadsheet)
    if int(run_parameters['verbose']) != 0:
        echo_input(np.zeros((1, 1)), spreadsheet, run_parameters)
    sample_names = ss_df.columns

    return spreadsheet, sample_names

def run_cc_net_nmf(run_parameters):
    """ Wrapper for call sequence that performs network based stratification
        with consensus clustering.

    Args:
        run_parameters: parameter set dictionary
    """
    adj_mat, spreadsheet, sample_names, lap_diag, lap_pos = get_net_nmf_input(run_parameters)

    form_and_save_h_clusters(adj_mat, spreadsheet, lap_diag, lap_pos, run_parameters)

    linkage_matrix, indicator_matrix = initialization(spreadsheet)
    consensus_matrix = form_consensus_matrix(
        run_parameters, linkage_matrix, indicator_matrix)
    labels = cluster_consensus_matrix(consensus_matrix, int(run_parameters['k']))

    save_cc_net_nmf_result(consensus_matrix, sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = form_consensus_matrix_graphic(consensus_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)
        
    return

def save_cc_net_nmf_result(consensus_matrix, sample_names, labels, run_parameters):
    """ write the results of consensus clustering network based nmt to output files

    Args:
        consensus_matrix: sample_names X labels symmetric consensus matrix
        sample_names: spreadsheet column names
        labels: cluster assignments for column names or consensus matrix
        run_parameters: python dictionary with "run_directory"
    """
    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    save_clusters(sample_names, labels, run_parameters)

    return

def create_df_with_sample_labels(sample_names, labels):
    """ create a dataframe with the spreadsheet column names as index and
        the cluster numbers as data

    Args:
        sample_names: spreadsheet column names
        labels: cluster number assignments

    Returns:
        clusters_dataframe:  pandas dataframe with paired sample_names and labels
    """
    clusters_dataframe = pd.DataFrame(data=labels, index=sample_names)

    return clusters_dataframe

def run_net_nmf(run_parameters):
    """ Wrapper for call sequence that performs network based stratification

    Args:
        run_parameters: parameter set dictionary
    """
    adj_mat, spreadsheet, sample_names, lap_diag, lap_pos = get_net_nmf_input(run_parameters)
    sample_smooth, iterations = smooth_spreadsheet_with_rwr(spreadsheet, adj_mat,
                                    np.float64(run_parameters["restart_probability"]))
    sample_quantile_norm = get_quantile_norm(sample_smooth)
    h_mat = perform_net_nmf(sample_quantile_norm, lap_pos, lap_diag, np.int_(run_parameters["k"]))

    sp_size = spreadsheet.shape[1]
    linkage_matrix = np.zeros((sp_size, sp_size))
    sample_perm = np.arange(0, sp_size)
    linkage_matrix = update_linkage_matrix(h_mat, sample_perm, linkage_matrix)
    labels = cluster_consensus_matrix(linkage_matrix, np.int_(run_parameters["k"]))

    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = form_consensus_matrix_graphic(linkage_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return
    
def run_cc_nmf(run_parameters):
    """ Wrapper for call sequence that performs non-negative matrix factorization
        with consensus clustering.

    Args:
        run_parameters: parameter set dictionary
    """
    spreadsheet, sample_names = get_nmf_input(run_parameters)
    nmf_form_save_h_clusters(spreadsheet, run_parameters)
    linkage_matrix, indicator_matrix = initialization(spreadsheet)
    consensus_matrix = form_consensus_matrix(
        run_parameters, linkage_matrix, indicator_matrix)
    labels = cluster_consensus_matrix(consensus_matrix, int(run_parameters['k']))
    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = form_consensus_matrix_graphic(consensus_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return

def run_nmf(run_parameters):
    """ Wrapper for call sequence that performs non-negative matrix factorization

    Args:
        run_parameters: parameter set dictionary
    """
    spreadsheet, sample_names = get_nmf_input(run_parameters)
    h_mat = nmf(spreadsheet, np.int_(run_parameters["k"]))
    sp_size = spreadsheet.shape[1]
    linkage_matrix = np.zeros((sp_size, sp_size))
    sample_perm = np.arange(0, sp_size)
    linkage_matrix = update_linkage_matrix(h_mat, sample_perm, linkage_matrix)
    labels = cluster_consensus_matrix(linkage_matrix, np.int_(run_parameters["k"]))
    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = form_consensus_matrix_graphic(linkage_matrix, int(run_parameters['k']))
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
    """
    for sample in range(0, np.int_(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = pick_a_sample(spreadsheet,
                                            np.float64(run_parameters["percent_sample"]))
        sample_smooth, iterations = \
        smooth_spreadsheet_with_rwr(sample_random, adj_mat,
                                    np.float64(run_parameters["restart_probability"]))

        if int(run_parameters['verbose']) != 0:
            print("{} of {}: iterations = {}".format(
                sample + 1,
                run_parameters["number_of_bootstraps"],
                iterations))

        sample_quantile_norm = get_quantile_norm(sample_smooth)
        h_mat = perform_net_nmf(sample_quantile_norm, lap_val, lap_dag,
                                np.int_(run_parameters["k"]))

        save_temporary_cluster(h_mat, sample_permutation, run_parameters, sample)

    return

def save_temporary_cluster(h_matrix, sample_permutation, run_parameters, sequence_number):
    """ save one h_matrix and its permutation in temorary files with
        sequence_number appended names
    Args:
        h_matrix: k x permutation sized encoding matrix
        sample_permutation: indices of h_matrix second dimension
        run_parameters: parmaeters including the "tmp_directory" name
        sequence_number: temporary file name suffix
    """
    tmp_dir = run_parameters["tmp_directory"]
    hname = os.path.join(tmp_dir, ('temp_h' + str(sequence_number)))
    h_matrix.dump(hname)
    pname = os.path.join(tmp_dir, ('temp_p' + str(sequence_number)))
    sample_permutation.dump(pname)

    return

def nmf_form_save_h_clusters(spreadsheet, run_parameters):
    """ Computes the components for the non-negative matric factorization
        consensus matrix from the input spreadsheet.

    Args:
        spreadsheet: genes x samples matrix
        run_parameters: dictionay of run-time parameters
    """
    for sample in range(0, np.int_(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = pick_a_sample(spreadsheet,
                                                np.float64(run_parameters["percent_sample"]))

        h_mat = nmf(sample_random, np.int_(run_parameters["k"]))
        save_temporary_cluster(h_mat, sample_permutation, run_parameters, sample)

        if int(run_parameters['verbose']) != 0:
            print('nmf {} of {}'.format(
                sample + 1, run_parameters["number_of_bootstraps"]))

    return

def form_indicator_matrix(run_parameters, indicator_matrix):
    """ read anonymous bootstrap tmp files compute the indicator_matrix for
        whichever method wrote them.

    Args:
        run_parameters: parameter set dictionary
        indicator_matrix: indicator matrix from initialization or previous

    Returns:
        indicator_matrix: indicator matrix summed with temp files permutations
    """
    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = np.int_(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        pname = os.path.join(tmp_dir, ('temp_p' + str(sample)))
        sample_permutation = np.load(pname)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)

    return indicator_matrix

def form_linkage_matrix(run_parameters, linkage_matrix):
    """ read anonymous bootstrap tmp files compute the linkage_matrix for
        whichever method wrote them.

    Args:
        run_parameters: parameter set dictionary
        linkage_matrix: connectivity matrix from initialization or previous

    Returns:
        linkage_matrix: linkage_matrix summed with linkages from temp files
    """
    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = np.int_(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        hname = os.path.join(tmp_dir, ('temp_h' + str(sample)))
        h_mat = np.load(hname)
        pname = os.path.join(tmp_dir, ('temp_p' + str(sample)))
        sample_permutation = np.load(pname)
        linkage_matrix = update_linkage_matrix(h_mat, sample_permutation, linkage_matrix)

    return linkage_matrix

def form_consensus_matrix(run_parameters, linkage_matrix, indicator_matrix):
    """ read anonymous bootstrap tmp files compute the consensus matrix for
        whichever method wrote them.

    Args:
        run_parameters: parameter set dictionary
        linkage_matrix: connectivity matrix from initialization or previous
        indicator_matrix: indicator matrix from initialization or previous

    Returns:
        consensus_matrix: sum of connectivity matrices / indicator matrices sum

    Removed:

    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = np.int_(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        hname = os.path.join(tmp_dir, ('temp_h' + str(sample)))
        h_mat = np.load(hname)
        pname = os.path.join(tmp_dir, ('temp_p' + str(sample)))
        sample_permutation = np.load(pname)
        linkage_matrix = update_linkage_matrix(h_mat, sample_permutation, linkage_matrix)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)
    """
    indicator_matrix = form_indicator_matrix(run_parameters, indicator_matrix)
    linkage_matrix = form_linkage_matrix(run_parameters, linkage_matrix)
    consensus_matrix = linkage_matrix / np.maximum(indicator_matrix, 1)

    return consensus_matrix

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

def pick_a_sample(spreadsheet, percent_sample):
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

def smooth_spreadsheet_with_rwr(restart, network_sparse, alpha=0.7, max_iteration=100, tol=1.e-4):
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

def get_quantile_norm(sample):
    """Normalizes an array using quantile normalization (ranking)

    Args:
        sample: initial sample

    Returns:
        sample_quantile_norm: quantile normalized sample
    """
    index = np.argsort(sample, axis=0)
    sample_sorted_by_rows = np.sort(sample, axis=0)
    mean_per_row = sample_sorted_by_rows.mean(1)
    sample_quantile_norm = sample.copy()
    for j in range(0, sample.shape[1]):
        sample_quantile_norm[index[:, j], j] = mean_per_row[:]

    return sample_quantile_norm

def get_h(w_matrix, x_matrix):
    """Finds a nonnegative right factor (H) of the perform_net_nmf function
    X ~ W.H

    Args:
        w_matrix: the positive left factor (W) of the perform_net_nmf function
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

def perform_net_nmf(x_matrix, lap_val, lap_dag, k=3, lmbda=1400, it_max=10000, h_clust_eq_limit=200,
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

    return h_matrix

def initialization(spreadsheet):
    ''' Initializes connectivity and indicator matrices.

    Args:
         spreadsheet: user's data

    Returns:
        linkage_matrix: samples x samples matrix of zeros
        indicator_matrix: samples x samples matrix of zeros
    '''
    sp_size = spreadsheet.shape[1]
    linkage_matrix = np.zeros((sp_size, sp_size))
    indicator_matrix = np.zeros((sp_size, sp_size))

    return  linkage_matrix, indicator_matrix

def update_linkage_matrix(encode_mat, sample_perm, linkage_matrix):
    '''Updates the connectivity matrix

    Args:
        encode_mat: nonnegative right factor (H)
        sample_perm: sample permutaion of h_matrix
        linkage_matrix: connectivity matrix

    Returns:
        linkage_matrix: modified connectivity matrix
    '''
    num_clusters = encode_mat.shape[0]
    cluster_id = np.argmax(encode_mat, 0)
    for cluster in range(0, num_clusters):
        slice_id = sample_perm[cluster_id == cluster]
        linkage_matrix[slice_id[:, None], slice_id] += 1
    return linkage_matrix

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

def cluster_consensus_matrix(consensus_matrix, k=3):
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

def form_consensus_matrix_graphic(consensus_matrix, k=3):
    '''Performs K-means and use its labels to reorder the consensus matrix

    Args:
        consensus_matrix: unordered consensus
        k: number of clusters

    Returns:
        cc_cm: consensus_matrix with rows and columns in k-means sort order
    '''
    cc_cm = consensus_matrix.copy()
    labels = cluster_consensus_matrix(consensus_matrix, k)
    sorted_labels = np.argsort(labels)
    cc_cm = cc_cm[sorted_labels[:, None], sorted_labels]

    return cc_cm

def echo_input(network, spreadsheet, run_parameters):
    ''' prints User's spread sheet and network data Dimensions and sizes

    Args:
         network: full gene-gene network
         spreadsheet: user's genes x samples data
         run_parameters: run parameters dictionary
    '''
    net_rows = network.shape[0]
    net_cols = network.shape[1]
    usr_rows = spreadsheet.shape[0]
    usr_cols = spreadsheet.shape[1]
    print('\nMethod: {}'.format(run_parameters['method']))
    date_frm = "Local: %a, %d %b %Y %H:%M:%S"
    print('Data Loaded:\t{}'.format(time.strftime(date_frm, time.localtime())))
    print('\nnetwork_file_name: {}'.format(run_parameters['network_file_name']))
    print('network    matrix {} x {}'.format(net_rows, net_cols))
    print('\nsamples_file_name: {}'.format(run_parameters['samples_file_name']))
    print('spread sheet matrix {} x {}\n'.format(usr_rows, usr_cols))
    print('\nAll run parameters as received:\n')
    display_run_parameters(run_parameters)

    return

def display_run_parameters(run_parameters):
    """ display the run parameters dictionary

    Args:
        run_parameters: python dictionary of run parameters
    """
    for fielap_dag_n in run_parameters:
        print('{} : {}'.format(fielap_dag_n, run_parameters[fielap_dag_n]))
    print('\n')

    return

def display_clusters(consensus_matrix):
    ''' display the consensus matrix.

    Args:
         consenus matrix: usually a smallish square matrix
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
    """ write the consensus matrix as a dataframe with column names and cluster
        labels as (non-unique) rows

    Args:
        consensus_matrix: labels x columns numerical matrix
        columns: data identifiers for column names
        labels: cluster numbers for row names
        run_parameters: contains directory path to write to consensus_data file
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = os.path.join(run_parameters["run_directory"], now_name('consensus_data', 'df'))
    else:
        file_name = os.path.join(run_parameters["run_directory"], 'consensus_data.df')
    out_df = pd.DataFrame(data=consensus_matrix, columns=columns, index=labels)
    out_df.to_csv(file_name, sep='\t')

    return

def save_clusters(sample_names, labels, run_parameters):
    """ wtite a two column file that attaches a cluster number to the sample names

    Args:
        sample_names: data identifiers (unique)
        labels: cluster number assignments (not unique)
        file_name: write path and file name
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = os.path.join(run_parameters["run_directory"], now_name('labels_data', 'tsv'))
    else:
        file_name = os.path.join(run_parameters["run_directory"], 'labels_data.tsv')

    df_tmp = pd.DataFrame(data=labels, index=sample_names)
    df_tmp.to_csv(file_name, sep='\t', header=None)

    return

def now_name(name_base, name_extension, time_step=1e6):
    """ insert a time stamp into the filename_ before .extension

    Args:
        name_base: file name first part - may include directory path
        name_extension: file extension without a period
        time_step: minimum time between two time stamps

    Returns:
        time_stamped_file_name: concatenation of the inputs with time-stamp
    """
    nstr = np.str_(np.int_(time.time() * time_step))
    time_stamped_file_name = name_base + '_' + nstr + '.' + name_extension

    return time_stamped_file_name

def run_parameters_dict():
    """ Dictionary of parameters: field names with default values. Also see
        module function generate_run_file to write parameters to txt file.

    Args: None

    Returns:
        run_parameters: a python dictionay of default parameters needed to run the
            functions in this module.
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
    """ Write a defaut parameter set to a text file for editing

    Args:
        file_name: file name (will be written as plain text).
    """
    par_dataframe = pd.DataFrame.from_dict(run_parameters_dict(), orient='index')
    par_dataframe.to_csv(file_name, sep='\t', header=False)

    return
