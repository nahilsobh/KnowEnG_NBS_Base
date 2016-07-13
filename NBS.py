"""
Created on Tue Jun 21 11:48:41 2016

@author: dlanier
@author: nahilsobh

This script performs network based clustering
"""

from knpackage import toolbox as kn
import numpy as np
import h5py
import argparse
import sys
  
def main(argv=None):
    '''Performs network based clustering'''
    if argv is None:
        argv = sys.argv
    else:
        argv = sys.argv.extend(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-keg_data','--keg_run_data',type=str,
                        help='-keg_data keg_run_data.hdf5 filename')
    parser.add_argument('-target_filename','--target_file',type=str,
                        help='-target_filename hdf5 filename (will overwrite)')
    args = parser.parse_args()
    
    network, spreadsheet, Ld, Lk, nbs_par_set = kn.get_keg_input(args.keg_run_data) 
    
    M, I = kn.consensus_cluster_nbs(network, spreadsheet, Ld, Lk, nbs_par_set)
                                    
    write_file = h5py.File(args.target_file, 'w')
    M_dataset = write_file.create_dataset('M',(M.shape),dtype=np.float64)
    M_dataset[...] = M
    I_dataset = write_file.create_dataset('I',(I.shape),dtype=np.float64)
    I_dataset[...] = I
    write_file.close()

if __name__ == "__main__":
    main()
