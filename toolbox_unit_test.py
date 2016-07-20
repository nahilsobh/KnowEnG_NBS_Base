# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 07:39:52 2016
knpackage.toolbox.py unit test

@author: del
"""

import unittest
import knpackage.toolbox as keg
import numpy as np
import scipy.sparse as spar
import scipy.linalg as sLA

class test_toolbox(unittest.TestCase):    
    
    def setUp(self):
        print('setup exec')
        
    def teardown(self):
        print('teardown exec')
        
    def testRWR(self):
        alpha = 1.0
        maxIt = 100
        tol = 1e-8

        F0 = np.array([[1.0], [0.0]])
        
        A = spar.csr_matrix(( (np.eye(2) + np.ones(2)) / 3.0) )
        # A = A.todense()
        F_exact = np.array([[0.5], [0.5]])
        F_calculated, niterz = keg.rwr(F0, A, alpha, maxIt, tol)
        mat_norm = F_exact - F_calculated
        mat_norm = np.sqrt((mat_norm * mat_norm).sum())
        self.assertLessEqual(mat_norm, tol)
        
    def testRWR_2(self):

        alpha = 0.5
        maxIt = 100
        tol = 1e-8
        F0 = np.array([[1.0], [0.0]])
        A = spar.csr_matrix(((np.eye(2) + np.ones(2)) / 3.0))   
        A = A.todense()
        F_exact = np.array([[0.8], [0.2]])
        F_calculated, niterz = keg.rwr(F0, A, alpha, maxIt, tol)
        mat_norm = (F_exact - F_calculated).T
        mat_norm = np.sqrt((mat_norm * mat_norm.T).sum())
        self.assertLessEqual(mat_norm, tol)     
        
    def testQN(self):
        a = np.array([[7.0, 5.0],[3.0, 1.0],[1.0,7.0]])
        aQN = np.array([[7.0, 4.0],[4.0,1.0],[1.0,7.0]])
        qn1 = keg.quantile_norm(a)
        
        self.assertEqual(sum(sum(qn1 != aQN)), 0, 'Quantile Norm 1 Not Equal')
        
    def test_form_network_laplacian(self):
        """ Ld, Lk = form_network_laplacian(adj_mat) """
        tst_mat = spar.csr_matrix(np.array(
            [[7,3,6,4],[3,7,4,5],[6,4,7,5],[4,5,5,7]]))
        tst_ans = np.array([[3,-1,-1,-1],[-1,3,-1,-1],[-1,-1,3,-1],[-1,-1,-1,3]])
        lap_dag, lap_val = keg.form_network_laplacian(tst_mat)
        graph_laplacian = lap_dag - lap_val
        self.assertEqual(((tst_ans - graph_laplacian) == True).sum(), 0)
        self.assertEqual((graph_laplacian.sum(axis=0) != 0).sum(), 0)
        self.assertEqual((graph_laplacian.sum(axis=1) != 0).sum(), 0)
            
    def test_normalized_matrix(self):
        """ adj_mat = normalized_matrix(adj_mat) """
        tol = 1e-1
        tst_mat = spar.csr_matrix(np.array(
            [[7,3,6,4],[3,7,4,5],[6,4,7,5],[4,5,5,7]]))
        tst_mat = tst_mat + tst_mat.T
        tst_mat_norm = keg.normalized_matrix(tst_mat)
        matrix_norm = sLA.norm(tst_mat_norm.todense())
        self.assertLessEqual(np.abs(matrix_norm - 1), tol, 'normalized_matrix')

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(test_toolbox))
    
    return test_suite
    
    
#if __name__=='__main__':
#    unittest.main()

"""                                        >> Preferred Method for using unit test
import unittest
import unit_test_toolbox as utkeg
mySuit = utkeg.suite()
runner = unittest.TextTestRunner()
myResult = runner.run(mySuit)

OR
mySuit2 = unittest.TestLoader().loadTestsFromTestCase(unit_test_toolbox)

"""   