
# from SPARTACUS.cli import main
import pytest

import numpy as np

import scipy.cluster.hierarchy as hierarchy

from SPARTACUS import spartacus as sp

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from SPARTACUS import spatial_silhouette as spasi

import sklearn.metrics as metrics


        
def test_SHAC_comparison_with_Python_function():
    """
    Camparsion with linkage function under no spatial constraint.
    """
    # Random input data with V = 24 variables on a 4x3x2 grid, where all 
    # variables are neighbors to each other, and ten subjects.
    V = 24
    X = np.random.normal(size = (10, V))
    matXYZ = np.zeros((V,3)) + 1
    vec_method = ["centroid", "median", "ward", "average", "single", "complete"]
    for method in vec_method:
        Z_allNeighbors = sp.shac(X, matXYZ, method = method, metric = 'euclidean',
                              diag_neighbor = False, standardize = False)
    
        Z_python = hierarchy.linkage(X.T, method = method, metric = "euclidean")
        if method == "ward":
            assert (np.abs(Z_allNeighbors[:,[0,1,3]] - Z_python[:,[0,1,3]]) < 0.0000000001).all(), "Different linkage matrix compared with hierarchy.linkage function (" + method + ")"
        else:
            assert (np.abs(Z_allNeighbors - Z_python) < 0.0000000001).all(), "Different linkage matrix compared with hierarchy.linkage function (" + method + ")"

def identical_labels(labels_true, labels_pred):
    """ 
    Calculates Rand index, Adjusted Rand index, Jaccard score and 
    Fowlkes-Mallows score.
    """
    uni_labels1 = np.unique(labels_true)
    uni_labels2 = np.unique(labels_pred)
    if uni_labels1.shape[0] != uni_labels2.shape[0]:
        return False
    for k in np.nditer(uni_labels1):
        if np.unique(labels_pred[np.where(labels_true == k)[0]]).shape[0] != 1:
            return False
    return True
        
    
def test_SHAC_comparison_with_Python_function_Ward():
    """
    Camparsion with AgglomerativeClustering package Ward.
    """
    vec_dim = (5,4,3) 
    connectivity = grid_to_graph(*vec_dim)
    n_clusters = 6
    # Random input data with V = 24 spatial variables on a 4x3x2 grid and ten 
    # subjects:
    V = np.prod(vec_dim)
    X = np.random.normal(size = (10, V))
    matXYZ = np.argwhere(np.zeros(vec_dim) == 0)
    # Compute clustering
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', affinity = "euclidean",
                                   connectivity=connectivity)
    ward.fit(X.T)
    Z = sp.shac(X, matXYZ, method = "ward", metric = 'euclidean',
                              diag_neighbor = False, standardize = False)
    labels_shac = sp.get_cluster(Z, V, n_init_cluster = n_clusters)
    assert identical_labels(labels_shac, ward.labels_), "SHAC Ward not identical to AgglomerativeClustering"


def test_SPARTACUS_comparison_with_R_function():
    """
    Camparsion with R function of clustering around latent components
    under no spatial constraint.
    """
    matrixA = np.genfromtxt('src/SPARTACUS/tests/matrixA.csv', delimiter=",", skip_header=1, usecols = range(1,21))
    matXYZ_A = np.zeros((matrixA.shape[1], 3))+1
    Z = sp.shac(matrixA, matXYZ_A, metric = 'spartacus', standardize = True)
    labelsA = sp.get_cluster(Z, matrixA.shape[1], n_init_cluster = 4)
    R_labelsA = np.genfromtxt('src/SPARTACUS/tests/R_labels_matrixA.csv', delimiter=",", skip_header=1).astype(int)[0,:]
    assert identical_labels(labelsA, R_labelsA), "Comparison with R function matrixA failed"
    

def is_spatial_contiguous(labels, list_neighbors):
    uni_labels = np.unique(labels)
    for k in np.nditer(uni_labels):
        id_voxels_k = np.where(labels == k)[0]
        if id_voxels_k.shape[0] == 1:
            continue
        list_neighbors_k = [list_neighbors[i] for i in id_voxels_k]
        all_neighbors_k = np.unique(np.concatenate(list_neighbors_k).ravel())
        if not set(id_voxels_k).issubset(set(all_neighbors_k)):
            return False
    return True
    
def test_spatial_contiguity():
    """
    Checks, whether SHAC methods generate spatially contiguous clusters.
    """
    vec_dim = (4,4,4) 
    V = np.prod(vec_dim)
    X = np.random.normal(size = (10, V))
    matXYZ = np.argwhere(np.zeros(vec_dim) == 0)
    list_neighbors = spasi.get_list_neighbors(matXYZ, diag_neighbor = False)
    vec_method = ["centroid", "median", "ward", "average", "single", "complete"]
    for method in vec_method:
        Z = sp.shac(X, matXYZ, method = method, metric = 'euclidean',
                    diag_neighbor = False, standardize = False)
        labels = sp.get_cluster(Z, V, n_init_cluster = 6)
        assert is_spatial_contiguous(labels, list_neighbors), "Not spatial contiguous (" + method + ")"
    Z = sp.shac(X, matXYZ, metric = 'spartacus',
                diag_neighbor = False, standardize = True)
    labels = sp.get_cluster(Z, V, n_init_cluster = 6)
    assert is_spatial_contiguous(labels, list_neighbors), "Not spatial contiguous (spartacus)"
        
    
def test_ensemble_simple_example_hellinger():
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
    assert (labels == np.array([1, 1, 3, 3, 2, 2, 4, 4])).all(), "Wrong labels"
    # Spatial contiguous?
    list_neighbors = spasi.get_list_neighbors(matXYZ, diag_neighbor = False)
    assert is_spatial_contiguous(labels, list_neighbors), "Not spatial contiguous (ensemble Hellinger)" 

def test_ensemble_simple_example_average():
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
    assert (labels == np.array([1, 1, 1, 1, 2, 2, 2, 2])).all(), "Wrong labels"
    # Spatial contiguous?
    list_neighbors = spasi.get_list_neighbors(matXYZ, diag_neighbor = False)
    assert is_spatial_contiguous(labels, list_neighbors), "Not spatial contiguous (ensemble average linkage)" 
    
def test_ensemble_simple_example_single():
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
    assert (labels == np.array([1, 1, 1, 1, 2, 2, 2, 2])).all(), "Wrong labels"    
    # Spatial contiguous?
    list_neighbors = spasi.get_list_neighbors(matXYZ, diag_neighbor = False)
    assert is_spatial_contiguous(labels, list_neighbors), "Not spatial contiguous (ensemble single linkage)" 
    
def tests_for_silhouette():
    """
    Does silhouette_coefficient() function produce the same results as 
    silhouette_score() function from sklearn.metrics using Euclidean metric?
    """
    # Test on matrixA
    X = np.genfromtxt('src/SPARTACUS/tests/matrixA.csv', delimiter=",", skip_header=1, usecols = range(1,21))
    V = X.shape[1]
    for i in range(3, 11):
        labels = np.random.randint(1, i+1, V)
        sil_score1 = spasi.silhouette_coefficient(X, labels, metric = "euclidean", iter_max = 10)    
        sil_score2 = metrics.silhouette_score(X.T, labels, metric = "euclidean")  
        assert np.round(sil_score1,10) == np.round(sil_score2, 10), "Silhouette function (Euclidean) produces different results than that implemented in scikit-learn"
    
    # Test on random data
    V = 100
    X = np.random.normal(size = (10, V))
    for i in range(3, 11):
        labels = np.random.randint(1, i+1, V)
        sil_score1 = spasi.silhouette_coefficient(X, labels, metric = "euclidean", iter_max = 10)    
        sil_score2 = metrics.silhouette_score(X.T, labels, metric = "euclidean")  
        assert np.round(sil_score1,10) == np.round(sil_score2, 10), "Silhouette function (Euclidean) produces different results than that implemented in scikit-learn"
    
    
def tests_for_ensemble_silhouette():
    X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])   
    labels = [1,1,2,2,3,3,4,4]
    assert spasi.silhouette_coefficient(X[0:4,], labels, metric = "jaccard", iter_max = 4) == 1, "Ensemble silhouette produces wrong results"
    sil_score1 = spasi.silhouette_coefficient(X, labels, metric = "jaccard", iter_max = 4)
    assert np.round(sil_score1, 8) == 0.79166667, "Ensemble silhouette produces wrong results"
    X1 = np.array([[1,1,2,2], [1,2,2,2], [1,1,1,2]])
    labels1 = [1,1,2,2]
    sil_score2 = spasi.silhouette_coefficient(X1, labels1, metric = "jaccard", iter_max = 4) 
    assert np.round(sil_score2, 8) == 0.46666667, "Ensemble silhouette produces wrong results"
    
# def test_spatial_silhouette():   
    
    
    
    
    
        
# def test_main():
#     assert main([]) == 0