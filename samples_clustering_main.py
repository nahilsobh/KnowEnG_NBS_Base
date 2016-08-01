# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:47:45 2016

@author: The Gene Sets Characterization dev team

"""

def nmf(run_parameters):
    '''nmf clustering'''
    from knpackage.toolbox import run_nmf
    run_nmf(run_parameters) 

def cc_nmf(run_parameters):
    '''kmeans consensus clustering of the nmf-based clusters'''
    from knpackage.toolbox import run_cc_nmf
    run_cc_nmf(run_parameters)

def net_nmf(run_parameters):
    '''net-nmf clustering "'''
    from knpackage.toolbox import run_net_nmf
    run_net_nmf(run_parameters)

def cc_net_nmf(run_parameters):
    '''kmeans consensus clustering of the net-nmf-based clusters'''
    from knpackage.toolbox import run_cc_net_nmf
    run_cc_net_nmf(run_parameters)

SELECT = {
    "cluster_nmf":nmf,
    "cc_cluster_nmf":cc_nmf,
    "net_cluster_nmf":net_nmf,
    "cc_net_cluster_nmf":cc_net_nmf}

def main():
    import sys
    from knpackage.toolbox import get_run_directory
    from knpackage.toolbox import get_run_parameters
    
    run_directory = get_run_directory(sys.argv)
    run_parameters = get_run_parameters(run_directory)
    SELECT[run_parameters["method"]](run_parameters)

if __name__ == "__main__":
    main()
