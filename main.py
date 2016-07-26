# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:47:45 2016

@author: The Gene Sets Characterization dev team
"""
import time

def nmf(run_parameters):
    t0 = time.time()
    print("method =", run_parameters["method"])
    from knpackage.toolbox import run_solo_nmf
    run_solo_nmf(run_parameters)
    print('\n\t\t\trun_nmf time {}'.format(time.time()-t0))
    
def cc_nmf(run_parameters):
    t0 = time.time()
    print("method =", run_parameters["method"])
    from knpackage.toolbox import run_cc_nmf
    run_cc_nmf(run_parameters)
    print('\n\t\t\tcc_nmf time {}'.format(time.time()-t0))
    
def nbsnmf(run_parameters):
    t0 = time.time()
    print("method =", run_parameters["method"])
    from knpackage.toolbox import run_solo_nbs
    run_solo_nbs(run_parameters)
    print('\n\t\t\trun_solo_nbs time {}'.format(time.time()-t0))
    
def cc_nbsnmf(run_parameters):
    t0 = time.time()
    print("method =", run_parameters["method"])
    from knpackage.toolbox import run_cc_nbs
    run_cc_nbs(run_parameters)
    print('\n\t\t\trun_cc_nbs time {}'.format(time.time()-t0))
    
# -------------------------------------
# map the inputs to the function blocks
# move to global_parameters
# -------------------------------------
SELECT = {
    "NMF": nmf,
    "NBS": nbsnmf,
    "cc_NMF": cc_nmf,
    "cc_NBS": cc_nbsnmf}

import sys
from knpackage.toolbox import get_input
from knpackage.toolbox import get_run_parameters

def main():
    # run main -work_dir /Users/del/KnowEnG_NBS_Local
    # --------------------------
    # Run the appropriate method found in RUN_FILE
    # --------------------------
    default_run_parameters_file='RUN_FILE'
    file = get_input(sys.argv, default_run_parameters_file)
    run_parameters = get_run_parameters(file)
    SELECT[run_parameters["method"]](run_parameters)

if __name__ == "__main__":
    main()