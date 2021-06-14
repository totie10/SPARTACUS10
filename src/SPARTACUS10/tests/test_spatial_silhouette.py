
# from SPARTACUS.cli import main
import pytest

import numpy as np

import pandas as pd

from SPARTACUS import spatial_silhouette as spasi

import sklearn.metrics as metrics

import os


def find_path(name, path = None):
    if path is None:
        path = os.getcwd()
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        
    
def test_silhouette():
    """
    Does silhouette_coefficient() function produce the same results as 
    silhouette_score() function from sklearn.metrics using Euclidean metric?
    """
    # Test on matrixA
    X = np.genfromtxt(find_path("matrixA.csv"), delimiter=",", skip_header=1, usecols = range(1,21))
    V = X.shape[1]
    for i in range(3, 11):
        labels = np.random.randint(1, i+1, V)
        sil_score1 = spasi.silhouette_coefficient(X, labels, metric = "euclidean", iter_max = 10)    
        sil_score2 = metrics.silhouette_score(X.T, labels, metric = "euclidean")  
        assert np.round(sil_score1,10) == np.round(sil_score2, 10), "Silhouette function (Euclidean) produces different results than that implemented in scikit-learn"
    
    # Test on random data comparison with existing function
    V = 100
    X = np.random.normal(size = (10, V))
    for i in range(3, 11):
        labels = np.random.randint(1, i+1, V)
        sil_score1 = spasi.silhouette_coefficient(X, labels, metric = "euclidean", iter_max = 10)    
        sil_score2 = metrics.silhouette_score(X.T, labels, metric = "euclidean")  
        assert np.round(sil_score1,10) == np.round(sil_score2, 10), "Silhouette function (Euclidean) produces different results than that implemented in scikit-learn"
    
    # Test on random data
    random_data = np.genfromtxt(find_path("random_data.csv"), delimiter=",")
    random_labels = np.genfromtxt(find_path("random_labels.csv"), delimiter=",")
    silhouette_score_Eucl = spasi.silhouette_coefficient(random_data, random_labels, metric = "euclidean")
    assert np.isclose(silhouette_score_Eucl, -0.018137954346288798), "Error in Euclidean silhouette_coefficient function"
    silhouette_score_corr = spasi.silhouette_coefficient(random_data, random_labels, metric = "correlation")
    assert np.isclose(silhouette_score_corr, -0.01710701512585803), "Error in correlation silhouette_coefficient function"
    
def test_ensemble_silhouette():
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
    
def test_simplified_silhouette():   
    # Test on random data
    random_data = np.genfromtxt(find_path("random_data.csv"), delimiter=",")
    random_labels = np.genfromtxt(find_path("random_labels.csv"), delimiter=",")
    simp_silhouette_score_Eucl = spasi.simplified_silhouette_coefficient(random_data, random_labels, metric = "euclidean")
    assert np.isclose(simp_silhouette_score_Eucl, 0.01761300723620632), "Error in Euclidean simplified_silhouette_coefficient function"
    simp_silhouette_score_corr = spasi.simplified_silhouette_coefficient(random_data, random_labels, metric = "correlation")
    assert np.isclose(simp_silhouette_score_corr, 0.07464102055366918), "Error in correlation simplified_silhouette_coefficient function"

def test_spatial_silhouette():
    # Test on random data
    random_data = np.genfromtxt(find_path("random_data_spatial.csv"), delimiter=",")
    matXYZ = np.argwhere(np.zeros((8, 3, 2)) == 0)
    labels = np.repeat(np.array([1,2,3,4]), 2*3*2)
    list_neighbors = spasi.get_list_neighbors(matXYZ) 
    spatial_silhouette_score_Eucl = spasi.silhouette_coefficient_spatial(random_data, labels, list_neighbors, metric = "euclidean")   
    assert np.isclose(spatial_silhouette_score_Eucl, -0.0019062813008068388), "Error in Euclidean silhouette_coefficient_spatial function"
    spatial_silhouette_score_corr = spasi.silhouette_coefficient_spatial(random_data, labels, list_neighbors, metric = "correlation")   
    assert np.isclose(spatial_silhouette_score_corr, -0.0013034499248535598), "Error in correlation silhouette_coefficient_spatial function"

def test_spatial_simplified_silhouette():
    # Test on random data
    random_data = np.genfromtxt(find_path("random_data_spatial.csv"), delimiter=",")
    matXYZ = np.argwhere(np.zeros((8, 3, 2)) == 0)
    labels = np.repeat(np.array([1,2,3,4]), 2*3*2)
    list_neighbors = spasi.get_list_neighbors(matXYZ) 
    spatial_simp_silhouette_score_Eucl = spasi.simplified_silhouette_coefficient_spatial(random_data, labels, list_neighbors, metric = "euclidean")   
    assert np.isclose(spatial_simp_silhouette_score_Eucl, 0.06783823739924444), "Error in Euclidean simplified_silhouette_coefficient_spatial function"
    spatial_simp_silhouette_score_corr = spasi.simplified_silhouette_coefficient_spatial(random_data, labels, list_neighbors, metric = "correlation")   
    assert np.isclose(spatial_simp_silhouette_score_corr, 0.22422765231602626), "Error in correlation simplified_silhouette_coefficient_spatial function"
    
def test_list_neighbors():
    list_neighbors_true = pd.read_csv(find_path("list_neighbors.csv"))
    list_neighbors_true.columns = pd.RangeIndex(start=0, stop=5, step=1)
    matXYZ = np.argwhere(np.zeros((4, 3, 2)) == 0)
    list_neighbors = spasi.get_list_neighbors(matXYZ)
    list_neighbors = pd.DataFrame(list_neighbors)
    list_neighbors.columns = pd.RangeIndex(start=0, stop=5, step=1)
    assert pd.DataFrame.equals(list_neighbors_true, list_neighbors), "list_neighbors does not work"
    # pd.testing.assert_frame_equal(list_neighbors_true, list_neighbors, check_dtype = False, check_column_type = False)

    
    
        
# def test_main():
#     assert main([]) == 0
