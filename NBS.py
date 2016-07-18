"""
Created on Tue Jun 21 11:48:41 2016

@author: dlanier
@author: nahilsobh


This script performs network based clustering
"""

from knpackage import toolbox as kn
import sys

def main():
    """ main function for finding clustering by network based stratification 
        with consensus clustering
    
    Args:
        defaults: knpackage.toolbox.nbs_par_set_dict
        args='-parameters /Users/del/tempOLE/keg_nbs_luad_h90q.df'
        args='-parameters /Users/del/tempOLE/keg_nbs_ov_h90q.df'
        args='-parameters /Users/del/tempOLE/keg_nbs_ucec_st90q.df'
    Returns:
        nothing
        
    """
    adj_mat, spreadsheet, par_set_dict, sample_names = kn.get_input(sys.argv)
    Ld, Lk = kn.form_network_laplacian(adj_mat)
    kn.form_and_save_h_clusters(adj_mat, spreadsheet, Ld, Lk, par_set_dict)
    connectivity_matrix, indicator_matrix = kn.initialization(spreadsheet)
    consensus_matrix = kn.retrieve_h_clusters_and_form_conensus_matrix(
        par_set_dict, connectivity_matrix, indicator_matrix)
    M, labels = kn.reorder_matrix(consensus_matrix, int(par_set_dict['k']))
    if int(par_set_dict['display_clusters']) != 0:
        kn.display_clusters(M)
    kn.write_sample_labels(sample_names, labels, par_set_dict['consensus_data_filename'])
    
if __name__ == "__main__":
    main()
