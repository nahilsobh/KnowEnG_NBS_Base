# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:47:45 2016

@author: The Gene Sets Characterization dev team
"""
# define the function blocks
def Fisher(session_parameters):
    print( "You typed Fisher\n")
    print("method =", session_parameters["method"])

def DRaWR(session_parameters):
    print( "You typed DRaWR\n")
    print("method =", session_parameters["method"])

def Net(session_parameters):
    print( "You typed Net\n")
    print("method =", session_parameters["method"])

# -------------------------------------
# map the inputs to the function blocks
# move to global_parameters
# -------------------------------------
SELECT = {"Fisher": Fisher, "DRaWR": DRaWR, "Net": Net}


from knpackage.toolbox import get_session_parameters 
# Dan Here.....
#def get_session_parameters():
#    session_parameters={"method":"Fisher"}
#    return session_parameters

# --------------------------
# Run the appropriate method
# --------------------------
file = "session_file"
session_parameters = get_session_parameters(file)
SELECT[session_parameters["method"]](session_parameters)