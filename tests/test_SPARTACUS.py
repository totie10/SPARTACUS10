
# from SPARTACUS.cli import main
import unittest

import numpy as np

from SPARTACUS import spartacus as sp

class TestSum(unittest.TestCase):
    def test_ensemble_simple_example_hellinger(self):
        """
        Tests example from help of spatial_ensemble_clustering().
        """
        X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])
        matXYZ = np.argwhere(np.zeros((2,2,2)) == 0)
        Z = sp.spatial_ensemble_clustering(X, matXYZ, method = "hellinger",
                         diag_neighbor = False)
        labels = sp.get_cluster(Z, V = 8, n_init_cluster = 4)
        self.assertEqual(labels, np.array([1, 1, 3, 3, 2, 2, 4, 4]))
    
    def test_ensemble_simple_example_average(self):
        """
        Tests example from help of spatial_ensemble_clustering().
        """
        X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])
        matXYZ = np.argwhere(np.zeros((2,2,2)) == 0)
        Z = sp.spatial_ensemble_clustering(X, matXYZ, method = "average",
                         diag_neighbor = False)
        labels = sp.get_cluster(Z, V = 8, n_init_cluster = 2)
        self.assertEqual(labels, np.array([1, 1, 1, 1, 2, 2, 2, 2]))
        
    def test_ensemble_simple_example_single(self):
        """
        Tests example from help of spatial_ensemble_clustering().
        """
        X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])
        matXYZ = np.argwhere(np.zeros((2,2,2)) == 0)
        Z = sp.spatial_ensemble_clustering(X, matXYZ, method = "single",
                         diag_neighbor = False)
        labels = sp.get_cluster(Z, V = 8, n_init_cluster = 2)
        self.assertEqual(labels, np.array([1, 1, 1, 1, 2, 2, 2, 2]))
        
    def test_SHAC_comparison_with_Python_function(self):
        """
        Camparsion with linkage function under no spatial constraint.
        """
    Z_allNeighbors = shac(X, matXYZ_allNeighbors, method='centroid', metric='euclidean',
                          diag_neighbor = False, standardize = False)
    
    Z_python = hierarchy.linkage(X.T, method = "centroid", metric = "euclidean")
    
    Z_allNeighbors - Z_python
        
    def test_SHAC_comparison_with_Python_function_Ward(self):
        """
        Camparsion with AgglomerativeClustering package Ward.
        """
        
    def test_SPARTACUS_comparison_with_R_function(self):
        """
        Camparsion with R function of clustering around latent components
        under no spatial constraint.
        """
        
    def test_spatial_contiguity(self):
        
        
# def test_main():
#     assert main([]) == 0
