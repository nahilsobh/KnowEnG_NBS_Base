"""
Created on Tue Jun 21 11:48:41 2016

@author: dlanier
@author: nahilsobh

This script performs network based clustering
"""

from knpackage import toolbox as kn
import scipy.sparse as spar
from numpy import maximum
  
def main():
    '''Performs network based clustering'''
    # ---------------------------------
    # Input, Clean, and set parameters
    # ---------------------------------
    network, spreadsheet, lut = kn.get_input()
    network_sparse = spar.csr_matrix(network)
    Ld, Lk = kn.form_network_laplacian(network)
    percent_sample, number_of_samples = kn.parameters_setup()
    network_size = network.shape[0]
    connectivity_matrix, indicator_matrix = kn.initialization(spreadsheet)

    # ----------------------------------------------
    # Network based clustering loop and aggregation
    # ----------------------------------------------
    for sample in range(0, number_of_samples):
        sample_random, sample_permutation = kn.get_a_sample(spreadsheet, percent_sample,
                                                         lut, network_size)
        sample_smooth, iterations = kn.rwr(sample_random, network_sparse, alpha=0.7)
        print("iterations = ", iterations)
        sample_quantile_norm = kn.quantile_norm(sample_smooth)
        H, niter = kn.netnmf(sample_quantile_norm, Lk, Ld, k=3)
        connectivity_matrix = kn.update_connectivity_matrix(H, sample_permutation, connectivity_matrix)
        indicator_matrix = kn.update_indicator_matrix(sample_permutation, indicator_matrix)
        
    # --------------------------
    # From the consensus matrix
    # --------------------------
    consensus_matrix = connectivity_matrix / maximum(indicator_matrix, 1)
    M = kn.reorder_matrix(consensus_matrix)
    
    # --------------------
    # Display the results
    # --------------------
    kn.display_clusters(M)

if __name__ == "__main__":
    main()
