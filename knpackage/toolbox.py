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
from numpy import maximum

import pandas as pd
import scipy.sparse as spar
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def get_run_directory_and_file(args):
    """ Read system input arguments (argv) to get the run directory name.

    Args:
        args: sys.argv, command line input; python main -run_directory dir_name

    Returns:
        run_directory: directory where run_file is expected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    parser.add_argument('-run_file', type=str)
    args = parser.parse_args()
    run_directory = args.run_directory
    run_file = args.run_file

    return run_directory, run_file

def get_run_parameters(run_directory, run_file):
    """ Read system input arguments run directory name and run_file into a dictionary.

    Args:
        run_directory: directory where run_file is expected.

    Returns:
        run_parameters: python dictionary of name - value parameters.
    """
    run_file_name = os.path.join(run_directory, run_file)
    par_set_df = pd.read_csv(run_file_name, sep='\t', header=None, index_col=0)
    run_parameters = par_set_df.to_dict()[1]
    run_parameters["run_directory"] = run_directory
    run_parameters["run_file"] = run_file

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

def get_network(network_name):
    """ Read in the cleaned subnet from KnowEnG network.

    Args:
        network_name: file name of cleaned network

    Returns:
        network_df: 3-column dataframe of cleaned network
    """
    network_df = pd.read_csv(
        network_name, header=None, names=None, delimiter='\t', usecols=[0, 1, 2])
    network_df.columns = ['node_1', 'node_2', 'wt']

    return network_df

def extract_network_node_names(network_df):
    """ extract node names lists from network.

    Args:
        netwrok_df: network dataframe.

    Returns:
        node_1_names: all names in column 1.
        node_list_2: all names in column 2.
    """
    node_list_1 = list(set(network_df.values[:, 0]))
    node_list_2 = list(set(network_df.values[:, 1]))

    return node_list_1, node_list_2

def find_unique_node_names(node_list_1, node_list_2):
    """ get the list (set union) of genes in either of the input lists.

    Args:
        node_list_1: list of node names.
        node_list_2: list of node names.

    Returns:
        unique_node_names: unique list of all node names.
    """
    unique_node_names = list(set(node_list_1) | set(node_list_2))

    return unique_node_names

def find_common_node_names(node_list_1, node_list_2):
    """ get the list (set intersection) of genes in both of the input lists.

    Args:
        node_list_1: list of node names.
        node_list_2: list of node names.

    Returns:
        common_node_names: unique list of common node names.
    """
    common_node_names = list(set(node_list_1) & set(node_list_2))

    return common_node_names

def extract_spreadsheet_gene_names(spreadsheet_df):
    """ get the uinque list (df.index.values) of genes in the spreadsheet dataframe.

    Args:
        spreadsheet_df: dataframe of spreadsheet input file.

    Returns:
        spreadsheet_gene_names: list of spreadsheet genes.
    """
    spreadsheet_gene_names = list(set(spreadsheet_df.index.values))

    return spreadsheet_gene_names

def write_spreadsheet_droplist(spreadsheet_df, unique_gene_names, run_parameters, file_name):
    """ write list of genes dropped from the input spreadsheed to
        run_parameters['tmp_directory'].file_name.

    Args:
        spreadsheet_df: the full spreadsheet data frame before dropping.
        unique_gene_names: the genes that will be used in calculation.
        run_parameters: dictionary of parameters.
        file_name: droped genes list file name.
    """
    tmp_dir = run_parameters['tmp_directory']
    droplist = spreadsheet_df.loc[~spreadsheet_df.index.isin(unique_gene_names)]
    file_path = os.path.join(tmp_dir, file_name)
    droplist = pd.DataFrame(droplist.index.values)
    droplist.to_csv(file_path, header=False, index=False)

    return

def update_spreadsheet(spreadsheet_df, gene_names):
    """ resize and reorder spreadsheet dataframe to only the gene_names list.

    Args:
        spreadsheet_df: dataframe of spreadsheet.
        unique_gene_names: list of all genes in network.

    Returns:
        spreadsheet_df: pandas dataframe of spreadsheet with only network genes.
    """
    updated_spreadsheet_df = spreadsheet_df.loc[gene_names].fillna(0)

    return updated_spreadsheet_df

def update_network(network, nodes_list, node_id):
    """ remove nodes not found as nodes_list in network node_id.

    Args:
        network: property to gene edges.
        intersection: user provided dataframe.

    Returns:
        updated_network: network that contains (rows) nodes_list found in node_id.
    """
    updated_network = network[network[node_id].isin(nodes_list)]

    return updated_network

def create_node_names_dictionary(node_names, start_value=0):
    """ create a python dictionary to look up gene locations from gene names

    Args:
        unique_gene_names: python list of gene names

    Returns:
        node_names_dictionary: python dictionary of gene names to integer locations
    """
    index_length = len(node_names) + start_value
    node_names_dictionary = dict(zip(node_names, np.arange(start_value, index_length)))

    return node_names_dictionary

def symmetrize_df(network):
    """ symmetrize network in sparse (3 cloumn) form.

    Args:
        network: property to gene edges.

    Returns:
        symm_network: symm_network[r, c] == symm_network[c, r], (network extended).
    """
    transpose = pd.DataFrame()
    transpose['node_1'] = network['node_2']
    transpose['node_2'] = network['node_1']
    transpose['wt'] = network['wt']
    symm_network = pd.concat([network, transpose])

    return symm_network

def map_node_names_to_index(network_df, genes_map, node_id):
    """ replace the node names with numbers for formation of numeric sparse matrix.

    Args:
        network_df: 3 col data frame version of network.
        genes_lookup_table: genes to location index dictionary.

    Returns:
        network_df: the same dataframe with integer indices in columns 0, 1.
    """
    network_df[node_id] = [genes_map[i] for i in network_df[node_id]]

    return network_df

def convert_df_to_sparse(network_df, matrix_length):
    """ network dataframe numerical columns [0,1,2] to scipy.sparse.csr_matrix.

    Args:
        network_df: padas dataframe with numerical columns [row_ix, col_ix, data].
        matrix_length: form size of square "network_sparse" matrix.

    Returns:
        network_sparse: scipy.sparse.csr_matrix.
    """
    nwm = network_df.as_matrix()
    network_sparse = spar.csr_matrix((np.float_(nwm[:, 2]),
                                      (np.int_(nwm[:, 0]), np.int_(nwm[:, 1]))),
                                     shape=(matrix_length, matrix_length))

    return network_sparse

def save_cc_net_nmf_result(consensus_matrix, sample_names, labels, run_parameters):
    """ write the results of network based nmf consensus clustering to output files.

    Args:
        consensus_matrix: sample_names x labels - symmetric consensus matrix.
        sample_names: spreadsheet column names.
        labels: cluster assignments for column names (or consensus matrix).
        run_parameters: dictionary with "results_directory" key.
    """
    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    save_clusters(sample_names, labels, run_parameters)

    return

def create_df_with_sample_labels(sample_names, labels):
    """ create dataframe from spreadsheet column names with cluster number assignments.

    Args:
        sample_names: spreadsheet column names.
        labels: cluster number assignments.

    Returns:
        clusters_dataframe: dataframe with sample_names keys to labels values.
    """
    clusters_dataframe = pd.DataFrame(data=labels, index=sample_names)

    return clusters_dataframe


def create_all_nodes_reverse_dict(dictionary):
    """ create reverse dictionary (keys > values, values > keys).

    Args:
        dictionary: dictionary.

    Returns:
        reverse dictionary: dictionary.
    """
    return {value: key for key, value in dictionary.items()}

def combime_dictionaries(dict1, dict2):
    """ combine two dictionaries into one.

    Args:
        dict1: dictionary.
        dict2: dictionary.

    Returns:
        combined_dictionary: dictionary.
    """
    return dict(dict1.items() + dict2.items())

def convert_network_df_to_sparse(pg_network_df, row_size, col_size):
    """  convert global network to sparse matrix.

    Args:
        pg_network_df: property-gene dataframe of global network (3 col)
        row_size: number of rows in sparse outpu
        col_size: number of columns in sparse outpu

    Returns:
        pg_network_sparse: sparse matrix of network gene set.
    """
    row_iden = pg_network_df.values[:, 1]
    col_iden = pg_network_df.values[:, 0]
    data = pg_network_df.values[:, 2]
    pg_network_sparse = spar.csr_matrix(
        (data, (row_iden, col_iden)), shape=(row_size, col_size))

    return pg_network_sparse

def perform_fisher_exact_test(
        sparse_matrix, property_idx_reverse_dict, user_set_input, universe_count, tmp_dir):
    """ cnetral loop: compute components for fisher exact test.

    Args:
        sparse_matrix: sparse matrix of network gene set.
        property_idx_reverse_dict: look up table of sparse matrix.
        user_set_input: the dataframe of user gene set.
        universe_count: count of the common_gene_names.
        tmp_dir: directory name to write results.
    """
    count = universe_count
    gene_count = sparse_matrix.sum(axis=0)
    df_val = []

    col_list = user_set_input.columns.values
    for col in col_list:
        new_user_set = user_set_input.loc[:, col]
        user_count = np.sum(new_user_set.values)
        overlap_count = sparse_matrix.T.dot(new_user_set.values)
        pval_overlap = np.zeros(len(property_idx_reverse_dict)) #pylint: disable=no-member
        for i, item_pval in enumerate(pval_overlap):
            table = build_contigency_table(overlap_count[i], user_count, gene_count[0, i], count)
            oddsratio, pvalue = stats.fisher_exact(table, alternative="greater")
            if overlap_count[i] != 0:
                row_item = [col, property_idx_reverse_dict[i], int(count),
                            int(user_count), int(gene_count[0, i]), int(overlap_count[i]), pvalue]
                df_val.append(row_item)
    df_col = ["user gene", "property", "count", "user count", "gene count", "overlap", "pval"]
    result_df = pd.DataFrame(df_val, columns=df_col).sort_values("pval", ascending=1)
    save_result(result_df, tmp_dir, "fisher_result.txt")

    return

def save_result(result_df, tmp_dir, file_name):
    """ save the result of DRaWR in tmp directory, file_name.

    Args:
        rw_result_df: dataframe of random walk result.
        tmp_dir: directory to save the result file.
        file_name: file name to save to.
    """
    file_path = os.path.join(tmp_dir, file_name)
    result_df.to_csv(file_path, header=True, index=False, sep='\t')

    return

def build_contigency_table(overlap_count, user_count, gene_count, count):
    """ build contigency table for fisher exact test.

    Args:
        overlap_count: count of overlaps in user gene set and network gene set.
        user_count: count of ones in user gene set.
        gene_count: count of ones in network gene set
        count: number of universe genes.

    Returns:
        table: the contigency table used in fisher test.
    """
    table = np.zeros(shape=(2, 2))
    table[0, 0] = overlap_count
    table[0, 1] = user_count - table[0, 0]
    table[1, 0] = gene_count - table[0, 0]
    table[1, 1] = count - user_count - gene_count + table[0, 0]

    return table

def run_DRaWR(run_parameters):
    spreadsheet_df = get_spreadsheet(run_parameters)
    pg_network_df  = get_network(run_parameters['pg_network_file_name'])
    gg_network_df  = get_network(run_parameters['gg_network_file_name'])

    pg_network_n1_names,\
    pg_network_n2_names = extract_network_node_names(pg_network_df)

    gg_network_n1_names,\
    gg_network_n2_names = extract_network_node_names(gg_network_df)

    # limit the gene set to the intersection of networks (gene_gene and prop_gene) and user gene set
    unique_gene_names     = find_unique_node_names(gg_network_n1_names, gg_network_n2_names)
    unique_gene_names     = find_unique_node_names(unique_gene_names, pg_network_n2_names)
    unique_all_node_names = unique_gene_names + pg_network_n1_names
   # unique_all_node_names = find_unique_gene_names(unique_gene_names, pg_network_n1_names)
    
    unique_gene_names_dict   = create_node_names_dictionary(unique_gene_names)
    pg_network_n1_names_dict = create_node_names_dictionary(pg_network_n1_names,len(unique_gene_names))

    # restrict spreadsheet to unique genes and drop everthing else
    spreadsheet_df = update_spreadsheet(spreadsheet_df, unique_all_node_names)
    # map every gene name to a sequential integer index
    gg_network_df = map_node_names_to_index(gg_network_df,unique_gene_names_dict, "node_1")
    gg_network_df = map_node_names_to_index(gg_network_df,unique_gene_names_dict, "node_2")
    pg_network_df = map_node_names_to_index(pg_network_df,pg_network_n1_names_dict, "node_1")
    pg_network_df = map_node_names_to_index(pg_network_df,unique_gene_names_dict, "node_2")
    
    gg_network_df = symmetrize_df(gg_network_df)
    pg_network_df = symmetrize_df(pg_network_df)
    
    gg_network_df = normalize_df(gg_network_df,'wt')
    pg_network_df = normalize_df(pg_network_df,'wt')
    
    hybrid_network_df = form_hybrid_network([gg_network_df, pg_network_df])
    
    # store the network in a csr sparse format
    network_sparse = convert_network_df_to_sparse(hybrid_network_df, len(unique_all_node_names), len(unique_all_node_names))
    
    perform_DRaWR(network_sparse, spreadsheet_df, len(unique_gene_names), run_parameters)


def perform_DRaWR(sparse_m, user_df, len_gene, run_parameters):
    """ calculate random walk with global network and user set gene sets  and write output.

    Args:
        sparse_m: sparse matrix of global network.
        user_df: dataframe of user gene sets.
        len_gene: length of genes in the in the user spreadsheet.
        run_parameters: parameters dictionary.
    """

    tmp_dir = run_parameters['results_directory']
    hetero_network = normalize(sparse_m, norm='l1', axis=0)
    new_user_df = append_baseline_to_spreadsheet(user_df, len_gene)
    new_user_matrix = normalize(new_user_df, norm='l1', axis=0)

    final_user_matrix, step = smooth_spreadsheet_with_rwr(
        new_user_matrix, hetero_network, run_parameters)
    final_user_df = pd.DataFrame(
        final_user_matrix, index=new_user_df.index.values, columns=new_user_df.columns.values)
    final_user_df = final_user_df.iloc[len_gene:]
    for col in final_user_df.columns.values[:-1]:
        final_user_df[col] = final_user_df[col] - final_user_df['base']
        final_user_df[col] = final_user_df.sort_values(col, ascending=0).index.values

    final_user_df['base'] = final_user_df.sort_values('base', ascending=0).index.values
    save_result(final_user_df, tmp_dir, "rw_result.txt")

    return

def append_baseline_to_spreadsheet(user_df, len_gene):
    """ append baseline vector of the user spreadsheet matrix.

    Args:
        user_df: user spreadsheet dataframe.
        len_gene: length of genes in the user spreadsheet.

    Returns:
        user_df: new dataframe with baseline vector appended in the last column.
    """
    property_size = user_df.shape[0] - len_gene
    user_df["base"] = np.append(np.ones(len_gene), np.zeros(property_size))

    return user_df

def normalize_df(network_df, node_id):
    """ normalize the network column with numbers for input.

    Args:
        network_df: network dataframe.
        node_id: column name

    Returns:
        network_df: the same dataframe with weight normalized.
    """
    network_df[node_id] /= network_df[node_id].sum()

    return network_df

def form_hybrid_network(list_of_networks):
    """ concatenate a list of networks.

    Args:
        list_of_networks: a list of networks to join

    Returns:
        a combined hybrid network
    """
    return pd.concat(list_of_networks)


def run_fisher(run_parameters):
    ''' fisher geneset characterization
    
    Args:
        run_parameters: dictionary of run parameters
    ''' 
    # -----------------------------------
    # - Data read and extractio Section -
    # -----------------------------------
    spreadsheet_df        = get_spreadsheet(run_parameters)
    prop_gene_network_df  = get_network(run_parameters['pg_network_file_name'])

    spreadsheet_gene_names     = extract_spreadsheet_gene_names(spreadsheet_df)

    prop_gene_network_n1_names,\
    prop_gene_network_n2_names = extract_network_node_names(prop_gene_network_df)

    # -----------------------------------------------------------------------
    # - limit the gene set to the intersection of network and user gene set -
    # -----------------------------------------------------------------------
    common_gene_names = find_common_node_names(prop_gene_network_n2_names, spreadsheet_gene_names)

    common_gene_names_dict                  = create_node_names_dictionary(common_gene_names)

    prop_gene_network_n1_names_dict         = create_node_names_dictionary(prop_gene_network_n1_names) 

    reverse_prop_gene_network_n1_names_dict = create_all_nodes_reverse_dict(prop_gene_network_n1_names_dict)

    # ----------------------------------------------------------------------------
    # - restrict spreadsheet and network to common genes and drop everthing else -
    # ----------------------------------------------------------------------------
    spreadsheet_df        = update_spreadsheet(spreadsheet_df, common_gene_names)
    prop_gene_network_df  = update_network(prop_gene_network_df, common_gene_names, "node_2")

    # ----------------------------------------------------------------------------
    # - map every gene name to an integer index in sequential order startng at 0 -
    # ----------------------------------------------------------------------------
    prop_gene_network_df = map_node_names_to_index(prop_gene_network_df, prop_gene_network_n1_names_dict, "node_1")
    prop_gene_network_df = map_node_names_to_index(prop_gene_network_df, common_gene_names_dict, "node_2")

    # --------------------------------------------
    # - store the network in a csr sparse format -
    # --------------------------------------------
    prop_gene_network_sparse = convert_network_df_to_sparse(prop_gene_network_df, len(common_gene_names),len(prop_gene_network_n1_names) )

    # ----------------------
    # - fisher exact test  -
    # ----------------------
    results_dir = run_parameters['results_directory']
    universe_count  = len(common_gene_names)
    perform_fisher_exact_test(prop_gene_network_sparse, reverse_prop_gene_network_n1_names_dict, spreadsheet_df, universe_count, results_dir)

    return

def run_nmf(run_parameters):
    """ wrapper: call sequence to perform non-negative matrix factorization and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    spreadsheet_df = get_spreadsheet(run_parameters)
    spreadsheet_mat = spreadsheet_df.as_matrix()
    spreadsheet_mat = get_quantile_norm(spreadsheet_mat)

    h_mat = nmf(spreadsheet_mat, run_parameters)

    linkage_matrix, indicator_matrix = initialization(spreadsheet_mat)
    sample_perm = np.arange(0, spreadsheet_mat.shape[1])
    linkage_matrix = update_linkage_matrix(h_mat, sample_perm, linkage_matrix)
    labels = kmeans_cluster_consensus_matrix(linkage_matrix, int(run_parameters['k']))

    sample_names = spreadsheet_df.columns
    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        con_mat_image = form_consensus_matrix_graphic(linkage_matrix, int(run_parameters['k']))
        display_clusters(con_mat_image)

    return

def run_cc_nmf(run_parameters):
    """ wrapper: call sequence to perform non-negative matrix factorization with
        consensus clustering and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    spreadsheet_df = get_spreadsheet(run_parameters)
    spreadsheet_mat = spreadsheet_df.as_matrix()
    spreadsheet_mat = get_quantile_norm(spreadsheet_mat)

    find_and_save_nmf_clusters(spreadsheet_mat, run_parameters)

    linkage_matrix, indicator_matrix = initialization(spreadsheet_mat)
    consensus_matrix = form_consensus_matrix(run_parameters, linkage_matrix, indicator_matrix)
    labels = kmeans_cluster_consensus_matrix(consensus_matrix, int(run_parameters['k']))

    sample_names = spreadsheet_df.columns
    write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters)
    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        display_clusters(form_consensus_matrix_graphic(consensus_matrix, int(run_parameters['k'])))

    return

def run_net_nmf(run_parameters):
    """ wrapper: call sequence to perform network based stratification and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    spreadsheet_df = get_spreadsheet(run_parameters)
    network_df = get_network(run_parameters['network_file_name'])

    node_1_names, node_2_names = extract_network_node_names(network_df)
    unique_gene_names = find_unique_node_names(node_1_names, node_2_names)
    genes_lookup_table = create_node_names_dictionary(unique_gene_names)

    network_df = map_node_names_to_index(network_df, genes_lookup_table, 'node_1')
    network_df = map_node_names_to_index(network_df, genes_lookup_table, 'node_2')

    network_df = symmetrize_df(network_df)
    network_mat = convert_df_to_sparse(network_df, len(unique_gene_names))

    network_mat = normalized_matrix(network_mat)
    lap_diag, lap_pos = form_network_laplacian(network_mat)

    spreadsheet_df = update_spreadsheet(spreadsheet_df, unique_gene_names)
    spreadsheet_mat = spreadsheet_df.as_matrix()
    sample_names = spreadsheet_df.columns

    sample_smooth, iterations = smooth_spreadsheet_with_rwr(
        spreadsheet_mat, network_mat, run_parameters)
    sample_quantile_norm = get_quantile_norm(sample_smooth)
    h_mat = perform_net_nmf(sample_quantile_norm, lap_pos, lap_diag, run_parameters)

    linkage_matrix, indicator_matrix = initialization(spreadsheet_mat)
    sample_perm = np.arange(0, spreadsheet_mat.shape[1])
    linkage_matrix = update_linkage_matrix(h_mat, sample_perm, linkage_matrix)
    labels = kmeans_cluster_consensus_matrix(linkage_matrix, int(run_parameters["k"]))

    save_clusters(sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        display_clusters(form_consensus_matrix_graphic(linkage_matrix, int(run_parameters['k'])))

    return

def run_cc_net_nmf(run_parameters):
    """ wrapper: call sequence to perform network based stratification with consensus clustering
        and write results.

    Args:
        run_parameters: parameter set dictionary.
    """
    spreadsheet_df = get_spreadsheet(run_parameters)
    network_df = get_network(run_parameters['network_file_name'])

    node_1_names, node_2_names = extract_network_node_names(network_df)
    unique_gene_names = find_unique_node_names(node_1_names, node_2_names)
    genes_lookup_table = create_node_names_dictionary(unique_gene_names)

    network_df = map_node_names_to_index(network_df, genes_lookup_table, 'node_1')
    network_df = map_node_names_to_index(network_df, genes_lookup_table, 'node_2')

    network_df = symmetrize_df(network_df)
    network_mat = convert_df_to_sparse(network_df, len(unique_gene_names))

    network_mat = normalized_matrix(network_mat)
    lap_diag, lap_pos = form_network_laplacian(network_mat)

    spreadsheet_df = update_spreadsheet(spreadsheet_df, unique_gene_names)
    spreadsheet_mat = spreadsheet_df.as_matrix()
    sample_names = spreadsheet_df.columns

    find_and_save_net_nmf_clusters(network_mat, spreadsheet_mat, lap_diag, lap_pos, run_parameters)

    linkage_matrix, indicator_matrix = initialization(spreadsheet_mat)
    consensus_matrix = form_consensus_matrix(
        run_parameters, linkage_matrix, indicator_matrix)
    labels = kmeans_cluster_consensus_matrix(consensus_matrix, int(run_parameters['k']))

    save_cc_net_nmf_result(consensus_matrix, sample_names, labels, run_parameters)

    if int(run_parameters['display_clusters']) != 0:
        display_clusters(form_consensus_matrix_graphic(consensus_matrix, int(run_parameters['k'])))

    return

def find_and_save_net_nmf_clusters(network_mat, spreadsheet_mat, lap_dag, lap_val, run_parameters):
    """ cnetral loop: compute components for the consensus matrix from the input
        network and spreadsheet matrices and save them to temp files.

    Args:
        network_mat: genes x genes symmetric matrix.
        spreadsheet_mat: genes x samples matrix.
        lap_dag, lap_val: laplacian matrix components; L = lap_dag - lap_val.
        run_parameters: dictionay of run-time parameters.
    """
    for sample in range(0, int(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = pick_a_sample(
            spreadsheet_mat, np.float64(run_parameters["percent_sample"]))
        sample_smooth, iterations = \
        smooth_spreadsheet_with_rwr(sample_random, network_mat, run_parameters)

        if int(run_parameters['verbose']) != 0:
            print("{} of {}: iterations = {}".format(
                sample + 1, run_parameters["number_of_bootstraps"], iterations))

        sample_quantile_norm = get_quantile_norm(sample_smooth)
        h_mat = perform_net_nmf(sample_quantile_norm, lap_val, lap_dag, run_parameters)

        save_temporary_cluster(h_mat, sample_permutation, run_parameters, sample)

    return

def find_and_save_nmf_clusters(spreadsheet_mat, run_parameters):
    """ cnetral loop: compute components for the consensus matrix by
        non-negative matrix factorization.

    Args:
        spreadsheet_mat: genes x samples matrix.
        run_parameters: dictionay of run-time parameters.
    """
    for sample in range(0, int(run_parameters["number_of_bootstraps"])):
        sample_random, sample_permutation = pick_a_sample(
            spreadsheet_mat, np.float64(run_parameters["percent_sample"]))

        h_mat = nmf(sample_random, run_parameters)
        save_temporary_cluster(h_mat, sample_permutation, run_parameters, sample)

        if int(run_parameters['verbose']) != 0:
            print('nmf {} of {}'.format(
                sample + 1, run_parameters["number_of_bootstraps"]))

    return

def save_temporary_cluster(h_matrix, sample_permutation, run_parameters, sequence_number):
    """ save one h_matrix and one permutation in temorary files with sequence_number appended names.

    Args:
        h_matrix: k x permutation size matrix.
        sample_permutation: indices of h_matrix columns permutation.
        run_parameters: parmaeters including the "tmp_directory" name.
        sequence_number: temporary file name suffix.
    """
    tmp_dir = run_parameters["tmp_directory"]
    hname = os.path.join(tmp_dir, ('temp_h' + str(sequence_number)))
    h_matrix.dump(hname)
    pname = os.path.join(tmp_dir, ('temp_p' + str(sequence_number)))
    sample_permutation.dump(pname)

    return

def form_indicator_matrix(run_parameters, indicator_matrix):
    """ read bootstrap temp_p files saved by central loop and compute the indicator_matrix.

    Args:
        run_parameters: parameter set dictionary.
        indicator_matrix: indicator matrix from initialization or previous call.

    Returns:
        indicator_matrix: input summed with "temp_p*" files in run_parameters["tmp_directory"].
    """
    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = int(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        pname = os.path.join(tmp_dir, ('temp_p' + str(sample)))
        sample_permutation = np.load(pname)
        indicator_matrix = update_indicator_matrix(sample_permutation, indicator_matrix)

    return indicator_matrix

def form_linkage_matrix(run_parameters, linkage_matrix):
    """ read bootstrap temp_h files compute the linkage_matrix.

    Args:
        run_parameters: parameter set dictionary.
        linkage_matrix: connectivity matrix from initialization or previous call.

    Returns:
        linkage_matrix: input summed with "temp_h*" files in run_parameters["tmp_directory"].
    """
    tmp_dir = run_parameters["tmp_directory"]
    number_of_bootstraps = int(run_parameters["number_of_bootstraps"])
    for sample in range(0, number_of_bootstraps):
        hname = os.path.join(tmp_dir, ('temp_h' + str(sample)))
        h_mat = np.load(hname)
        pname = os.path.join(tmp_dir, ('temp_p' + str(sample)))
        sample_permutation = np.load(pname)
        linkage_matrix = update_linkage_matrix(h_mat, sample_permutation, linkage_matrix)

    return linkage_matrix

def form_consensus_matrix(run_parameters, linkage_matrix, indicator_matrix):
    """ compute the consensus matrix from the indicator and linkage matrix inputs
        and the "temp_*" files stored in run_parameters["tmp_directory"].

    Args:
        run_parameters: parameter set dictionary with "tmp_directory" key.
        linkage_matrix: linkage matrix from initialization or previous call.
        indicator_matrix: indicator matrix from initialization or previous call.

    Returns:
        consensus_matrix: (sum of linkage matrices) / (sum of indicator matrices).
    """
    indicator_matrix = form_indicator_matrix(run_parameters, indicator_matrix)
    linkage_matrix = form_linkage_matrix(run_parameters, linkage_matrix)
    consensus_matrix = linkage_matrix / np.maximum(indicator_matrix, 1)

    return consensus_matrix

def normalized_matrix(network_mat):
    """ square root of inverse of diagonal D (D * network_mat * D) normaization.

    Args:
        network_mat: symmetric matrix.

    Returns:
        network_mat: input matrix - renomralized s.t sum of row or col ~= 1.
    """
    row_sm = np.array(network_mat.sum(axis=0))
    row_sm = 1.0 / row_sm
    row_sm = np.sqrt(row_sm)
    r_c = np.arange(0, network_mat.shape[0])
    diag_mat = spar.csr_matrix((row_sm[0, :], (r_c, r_c)), shape=(network_mat.shape))
    network_mat = diag_mat.dot(network_mat)
    network_mat = network_mat.dot(diag_mat)

    return network_mat

def form_network_laplacian(network_mat):
    """ Laplacian matrix components for use in network based stratification.

    Args:
        network_mat: symmetric matrix.

    Returns:
        diagonal_laplacian: diagonal of the laplacian matrix.
        laplacian: locations in the laplacian matrix.
    """
    laplacian = spar.lil_matrix(network_mat.copy())
    laplacian.setdiag(0)
    laplacian[laplacian != 0] = 1
    diag_length = laplacian.shape[0]
    rowsum = np.array(laplacian.sum(axis=0))
    diag_arr = np.arange(0, diag_length)
    diagonal_laplacian = spar.csr_matrix((rowsum[0, :], (diag_arr, diag_arr)),
                                         shape=(network_mat.shape))
    laplacian = laplacian.tocsr()

    return diagonal_laplacian, laplacian

def pick_a_sample(spreadsheet_mat, percent_sample):
    """ percent_sample x percent_sample random sample, from spreadsheet_mat.

    Args:
        spreadsheet_mat: gene x sample spread sheet as matrix.
        percent_sample: decimal fraction (slang-percent) - [0 : 1].

    Returns:
        sample_random: A specified precentage sample of the spread sheet.
        sample_permutation: the array that correponds to random sample.
    """
    features_size = int(np.round(spreadsheet_mat.shape[0] * (1-percent_sample)))
    features_permutation = np.random.permutation(spreadsheet_mat.shape[0])
    features_permutation = features_permutation[0:features_size].T

    patients_size = int(np.round(spreadsheet_mat.shape[1] * percent_sample))
    sample_permutation = np.random.permutation(spreadsheet_mat.shape[1])
    sample_permutation = sample_permutation[0:patients_size]

    sample_random = spreadsheet_mat[:, sample_permutation]
    sample_random[features_permutation[:, None], :] = 0

    positive_col_set = sum(sample_random) > 0
    sample_random = sample_random[:, positive_col_set]
    sample_permutation = sample_permutation[positive_col_set]

    return sample_random, sample_permutation

def smooth_spreadsheet_with_rwr(restart, network_sparse, run_parameters):
    """ simulate a random walk with restart. iterate: (R_n+1 = a*N*R_n + (1-a)*R_n).

    Args:
        restart: restart array of any column size.
        network_sparse: network stored in sparse format.
        run_parameters: parameters dictionary with "restart_probability",
        "restart_tolerance", "number_of_iteriations_in_rwr".

    Returns:
        smooth_1: smoothed restart data.
        step: number of iterations (converged to tolerence or quit).
    """
    tol = np.float_(run_parameters["restart_tolerance"])
    alpha = np.float_(run_parameters["restart_probability"])
    smooth_0 = restart
    smooth_r = (1. - alpha) * restart
    for step in range(0, int(run_parameters["number_of_iteriations_in_rwr"])):
        smooth_1 = alpha * network_sparse.dot(smooth_0) + smooth_r
        deltav = LA.norm(smooth_1 - smooth_0, 'fro')
        if deltav < tol:
            break
        smooth_0 = smooth_1

    return smooth_1, step

def get_quantile_norm(sample):
    """ normalizes an array using quantile normalization (ranking).

    Args:
        sample: initial sample - spreadsheet matrix.

    Returns:
        sample_quantile_norm: quantile normalized spreadsheet matrix.
    """
    index = np.argsort(sample, axis=0)
    sample_sorted_by_rows = np.sort(sample, axis=0)
    mean_per_row = sample_sorted_by_rows.mean(1)
    sample_quantile_norm = sample.copy()
    for j in range(0, sample.shape[1]):
        sample_quantile_norm[index[:, j], j] = mean_per_row[:]

    return sample_quantile_norm

def get_h(w_matrix, x_matrix):
    """ nonnegative right factor matrix for perform_net_nmf function s.t. X ~ W.H.

    Args:
        w_matrix: the positive left factor (W) of the perform_net_nmf function.
        x_matrix: the postive matrix (X) to be decomposed.

    Returns:
        h_matrix: nonnegative right factor (H) matrix.
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

def perform_net_nmf(x_matrix, lap_val, lap_dag, run_parameters):
    """ perform network based nonnegative matrix factorization, minimize:
        ||X-WH|| + lambda.tr(W'.L.W), with W, H positive.

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W.H
        lap_val: the laplacian matrix
        lap_dag: the diagonal of the laplacian matrix
        run_parameters: parameters dictionary with keys: "k", "lambda", "it_max",
            "h_clust_eq_limit", "obj_fcn_chk_freq".

    Returns:
        h_matrix: nonnegative right factor (H) matrix.
    """
    k = float(run_parameters["k"])
    lmbda = float(run_parameters["lmbda"])
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, int(run_parameters["it_max"])):
        if np.mod(itr, int(run_parameters["obj_fcn_chk_freq"])) == 0:
            h_clusters = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(h_clust_eq != h_clusters) == 0):
                h_eq_count = h_eq_count + int(run_parameters["obj_fcn_chk_freq"])
            else:
                h_eq_count = 0
            h_clust_eq = h_clusters
            if h_eq_count >= float(run_parameters["h_clust_eq_limit"]):
                break
        numerator = maximum(np.dot(x_matrix, h_matrix.T) + lmbda * lap_val.dot(w_matrix), epsilon)
        denomerator = maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T))
                              + lmbda * lap_dag.dot(w_matrix), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = get_h(w_matrix, x_matrix)

    return h_matrix

def nmf(x_matrix, run_parameters):
    """ nonnegative matrix factorization, minimize the diffence between X and W dot H
        with positive factor matrices W, and H.

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W dot H.
        run_parameters: parameters dictionary with keys "k", "it_max",
            "cluster_min_repeats", "obj_fcn_chk_freq".

    Returns:
        h_matrix: nonnegative right factor matrix (H).
    """
    k = float(run_parameters["k"])
    obj_fcn_chk_freq = int(run_parameters["obj_fcn_chk_freq"])
    h_clust_eq_limit = float(run_parameters["h_clust_eq_limit"])
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, int(run_parameters["it_max"])):
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

def initialization(spreadsheet_mat):
    ''' Initialize connectivity and indicator matrices size of spreadsheet columns.

    Args:
         spreadsheet_mat: user data input genes-rows, sample_names-columns.

    Returns:
        linkage_matrix: samples x samples matrix of zeros.
        indicator_matrix: samples x samples matrix of zeros.
    '''
    sp_size = spreadsheet_mat.shape[1]
    linkage_matrix = np.zeros((sp_size, sp_size))
    indicator_matrix = np.zeros((sp_size, sp_size))

    return  linkage_matrix, indicator_matrix

def update_linkage_matrix(encode_mat, sample_perm, linkage_matrix):
    ''' update the connectivity matrix by summing the un-permuted linkages.

    Args:
        encode_mat: (permuted) nonnegative right factor matrix (H) - encoded linkage.
        sample_perm: the sample permutaion of the h_matrix.
        linkage_matrix: connectivity matrix.

    Returns:
        linkage_matrix: connectivity matrix summed with the de-permuted linkage.
    '''
    num_clusters = encode_mat.shape[0]
    cluster_id = np.argmax(encode_mat, 0)
    for cluster in range(0, num_clusters):
        slice_id = sample_perm[cluster_id == cluster]
        linkage_matrix[slice_id[:, None], slice_id] += 1
    return linkage_matrix

def update_indicator_matrix(sample_perm, indicator_matrix):
    ''' update the indicator matrix by summing the un-permutation.

    Args:
        sample_perm: permutaion of the sample (h_matrix).
        indicator_matrix: indicator matrix.

    Returns:
        indicator_matrix: indicator matrix incremented at sample_perm locations.
    '''
    indicator_matrix[sample_perm[:, None], sample_perm] += 1

    return indicator_matrix

def kmeans_cluster_consensus_matrix(consensus_matrix, k=3):
    """ determine cluster assignments for consensus matrix using K-means.

    Args:
        consensus_matrix: connectivity / indicator matrix.
        k: clusters estimate.

    Returns:
        lablels: ordered cluster assignments for consensus_matrix (samples).
    """
    cluster_handle = KMeans(k, random_state=10)
    labels = cluster_handle.fit_predict(consensus_matrix)

    return labels

def form_consensus_matrix_graphic(consensus_matrix, k=3):
    ''' use K-means to reorder the consensus matrix for graphic display.

    Args:
        consensus_matrix: calculated consensus matrix in samples x samples order.
        k: number of clusters estimate (inner diminsion k of factored h_matrix).

    Returns:
        cc_cm: consensus_matrix with rows and columns in K-means sort order.
    '''
    cc_cm = consensus_matrix.copy()
    labels = kmeans_cluster_consensus_matrix(consensus_matrix, k)
    sorted_labels = np.argsort(labels)
    cc_cm = cc_cm[sorted_labels[:, None], sorted_labels]

    return cc_cm

def echo_input(network_mat, spreadsheet_mat, run_parameters):
    ''' command line display data: network and spreadsheet matrices and run parameters.

    Args:
         network_mat: gene-gene network matrix.
         spreadsheet_mat: genes x samples user spreadsheet data matrix.
         run_parameters: run parameters dictionary.
    '''
    net_rows = network_mat.shape[0]
    net_cols = network_mat.shape[1]
    usr_rows = spreadsheet_mat.shape[0]
    usr_cols = spreadsheet_mat.shape[1]
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
    """ command line display the run parameters dictionary.

    Args:
        run_parameters: dictionary of run parameters.
    """
    for fielap_dag_n in run_parameters:
        print('{} : {}'.format(fielap_dag_n, run_parameters[fielap_dag_n]))
    print('\n')

    return

def display_clusters(consensus_matrix):
    ''' graphic display the consensus matrix.

    Args:
         consenus matrix: usually a smallish square matrix.
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

def write_consensus_matrix(consensus_matrix, sample_names, labels, run_parameters):
    """ write the consensus matrix as a dataframe with sample_names column lablels
        and cluster labels as row labels.

    Args:
        consensus_matrix: sample_names x sample_names numerical matrix.
        sample_names: data identifiers for column names.
        labels: cluster numbers for row names.
        run_parameters: path to write to consensus_data file (run_parameters["results_directory"]).
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = os.path.join(
            run_parameters["results_directory"], now_name('consensus_data', 'df'))
    else:
        file_name = os.path.join(run_parameters["results_directory"], 'consensus_data.df')
    out_df = pd.DataFrame(data=consensus_matrix, columns=sample_names, index=labels)
    out_df.to_csv(file_name, sep='\t')

    return

def save_clusters(sample_names, labels, run_parameters):
    """ wtite .tsv file that assings a cluster number label to the sample_names.

    Args:
        sample_names: (unique) data identifiers.
        labels: cluster number assignments.
        run_parameters: write path (run_parameters["results_directory"]).
    """
    if int(run_parameters["use_now_name"]) != 0:
        file_name = os.path.join(
            run_parameters["results_directory"], now_name('labels_data', 'tsv'))
    else:
        file_name = os.path.join(run_parameters["results_directory"], 'labels_data.tsv')

    df_tmp = pd.DataFrame(data=labels, index=sample_names)
    df_tmp.to_csv(file_name, sep='\t', header=None)

    return

def now_name(name_base, name_extension, time_step=1e6):
    """ insert a time stamp into the filename_ before .extension.

    Args:
        name_base: file name first part - may include directory path.
        name_extension: file extension without a period.
        time_step: minimum time between two time stamps.

    Returns:
        time_stamped_file_name: concatenation of the inputs with time-stamp.
    """
    nstr = np.str_(int(time.time() * time_step))
    time_stamped_file_name = name_base + '_' + nstr + '.' + name_extension

    return time_stamped_file_name

def cluster_parameters_dict():
    """ dictionary of parameters: keys with default values.

    Args: None.

    Returns:
        run_parameters: dictionay of default key - values to run functions in this module.
    """
    run_parameters = {
        "method":"cc_net_cluster",
        "k":4,
        "number_of_bootstraps":5,
        "percent_sample":0.8,
        "restart_probability":0.7,
        "number_of_iteriations_in_rwr":100,
        "it_max":10000,
        "h_clust_eq_limit":200,
        "obj_fcn_chk_freq":50,
        "restart_tolerance":1e-4,
        'lmbda':1400,
        "network_file_name":"network_file_name",
        "samples_file_name":"samples_file_name",
        "tmp_directory":"tmp",
        "results_directory":"results",
        "use_now_name":1,
        "verbose":1,
        "display_clusters":1,
        'method1':'cluster_nmf',
        'method2':'cc_cluster_nmf',
        'method3':'net_cluster_nmf',
        'method4':'cc_net_cluster_nmf'}

    return run_parameters

def generate_run_file(run_parameters=cluster_parameters_dict(), file_name='run_file'):
    """ write a parameter set dictionary to a text file for editing.

    Args:
        file_name: file name (will be written as plain text).
    """
    par_dataframe = pd.DataFrame.from_dict(run_parameters, orient='index')
    par_dataframe.to_csv(file_name, sep='\t', header=False)

    return
