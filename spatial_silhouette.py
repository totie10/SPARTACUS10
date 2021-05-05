# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:49:19 2019

@author: admin
"""

import numpy as np
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA

def silhouette_width_index(X, labels, metric = "euclidean", iter_max = 20000,
                           return_vec_sil = False):
    """
    Calculates silhouette coefficient. For large data sets, pairwise distance 
    calculation is performed in portions. Therefore, in contrast to, e.g., 
    sklearn.metrics.silhouette_score, this function does not run into a memory
    error, if the number of objects to be clustered is very large. 

    Parameters
    ----------
    X : ndarray, shape(N, V)
        Input matrix including, e.g., grey matter volume values, where N is 
        the number of subjects and V is the number of variables, e.g. voxels. 
        Note that the cluster objects are the variables and not the subjects.
    labels : ndarray, shape (V,)
        Array of integers, indicating the cluster labels of the variables.
        Sorted as in the column of X, i.e. the j-th cluster label corresponds 
        to the variable stored in the j-th column of X.
    metric : str, optional
        The distance metric between two variables. It is recommended to either
        employ 'euclidean' (Euclidean distance) or 'correlation' (1-abs(corr)).
        However, this function technically allows all distance metrics as 
        implemented in ?scipy.spatial.distance.pdist. In the latter case, it is 
        up to the user to ensure the validity of the chosen metric. The default 
        is "euclidean".
    iter_max : int, optional
        In order to avoid a memory error, the pairwise variable distance 
        calculation is partitioned, if at least one of the two clusters includes 
        more than iter_max variables. The default is 20000.
    return_vec_sil : bool, optional
        Should the silhouette width of all single variables be returned in 
        addition to the silhouette coefficient? The default is False.

    Returns
    -------
    silhouette_coefficient : scalar
        The silhouette coefficient.
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the silhouette width of all single variables 
        is additionally returned.
        
    """
    # Determine labels indicating different clusters
    unique_labels = np.unique(labels)
    # Initiate vec_s
    vec_s = np.array([])
    # For each cluster
    for l in np.nditer(unique_labels):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == l)[0]
        # Get number of voxels in Cluster l
        n_voxels_l = id_vox_l.shape[0]
        # Calculate data matrix of cluster l
        if id_vox_l.shape[0] == 1:
            vec_s = np.append(vec_s, 0)
            continue
        else:
            X_l = X[:, id_vox_l]
        #### Divide voxels in cluster l in groups of maximum size = iter_max.
        # Get number of groups
        n_groups_l = int(np.ceil(np.array([n_voxels_l / iter_max])))
        # Initiate vec_a
        vec_a = np.zeros(n_voxels_l)
        # Compare each group with each group
        for i in range(n_groups_l):
            i_from = (i * iter_max)
            i_to = (np.min(((i + 1) * iter_max, n_voxels_l)))
            X_l_i = X_l[:, i_from:i_to]
            if metric == "correlation":
                tmp_mat = distance.squareform(distance.pdist(X_l_i.T, metric = metric))
                vec_scores = np.sum(1 - np.abs(1-tmp_mat), 0)
            else:
                vec_scores = np.sum(distance.squareform(distance.pdist(X_l_i.T, metric = metric)),0)
            vec_a[i_from:i_to] = vec_a[i_from:i_to] + vec_scores
            for j in range(i+1, n_groups_l):
                j_from = (j * iter_max)
                j_to = (np.min(((j + 1) * iter_max, n_voxels_l)))
                X_l_j = X_l[:, j_from:j_to]
                mat_dist_ij = distance.cdist(X_l_i.T, X_l_j.T, metric = metric)
                if metric == "correlation":
                    mat_dist_ij = 1 - np.abs(1-mat_dist_ij)
                vec_a[i_from:i_to] = vec_a[i_from:i_to] + np.sum(mat_dist_ij, 1)
                vec_a[j_from:j_to] = vec_a[j_from:j_to] + np.sum(mat_dist_ij, 0)
        # Calculate vector with within distances 
        vec_a = vec_a/(n_voxels_l - 1)
        #### Calculate vec_b
        # Initiate mat_b to later calculate column min from
        mat_b = np.zeros((unique_labels.shape[0] - 1, n_voxels_l))
        # Initiate counter
        counter = 0
        for h in np.nditer(unique_labels):
            if h == l:
                continue
            # ID voxels belongig to cluster h
            id_vox_h = np.where(labels == h)[0]
            # Get number of voxels in Cluster h
            n_voxels_h = id_vox_h.shape[0]
            # Calculate data matrix of cluster h
            if n_voxels_h == 1:
                X_h = X[:, id_vox_h].reshape((X.shape[0], 1))
            else:
                X_h = X[:, id_vox_h] 
            # Get number of groups for cluster h
            n_groups_h = int(np.ceil(np.array([n_voxels_h / iter_max])))
            # Compare each group with each group
            for i in range(n_groups_l):
                i_from = (i * iter_max)
                i_to = (np.min(((i + 1) * iter_max, n_voxels_l)))
                X_l_i = X_l[:, i_from:i_to]
                for j in range(n_groups_h):
                    j_from = (j * iter_max)
                    j_to = (np.min(((j + 1) * iter_max, n_voxels_h)))
                    X_h_j = X_h[:, j_from:j_to]
                    if metric == "correlation":
                        tmp_mat = distance.cdist(X_l_i.T, X_h_j.T, metric = metric)
                        vec_scores = np.sum(1 - np.abs(1-tmp_mat), 1)
                    else:
                        vec_scores = np.sum(distance.cdist(X_l_i.T, X_h_j.T, metric = metric), 1)
                    mat_b[counter, i_from:i_to] = mat_b[counter, i_from:i_to] + vec_scores
            # Calculate matrix with between distances
            mat_b[counter, :] = mat_b[counter, :] / n_voxels_h
            # Update counter
            counter += 1
        # Calculate vector with between distance
        vec_b = np.min(mat_b, 0)
        # Calculate s
        vec_s_l = (vec_b - vec_a) / np.maximum(vec_a, vec_b)
        # Update vec_s
        vec_s = np.append(vec_s, vec_s_l)
    # Return silhouette index
    silhouette_coefficient = np.mean(vec_s)
    if return_vec_sil:
        return silhouette_coefficient, vec_s
    else:
        return silhouette_coefficient

def simplified_silhouette_width_index(X, labels, metric = "euclidean",
                                      return_vec_sil = False):
    """
    Calculates simplified silhouette coefficient. For large data sets, pairwise 
    distance calculation is performed in portions. Therefore, in contrast to, e.g., 
    sklearn.metrics.silhouette_score, this function does not run into a memory
    error, if the number of objects to be clustered is very large. 

    Parameters
    ----------
    X : ndarray, shape(N, V)
        Input matrix including, e.g., grey matter volume values, where N is 
        the number of subjects and V is the number of variables, e.g. voxels. 
        Note that the cluster objects are the variables and not the subjects.
    labels : ndarray, shape (V,)
        Array of integers, indicating the cluster labels of the variables.
        Sorted as in the column of X, i.e. the j-th cluster label corresponds 
        to the variable stored in the j-th column of X.
    metric : str, optional
        The distance metric between two variables. One of 'euclidean' 
        (Euclidean distance) or 'correlation' (1-abs(corr)). The default is 
        "euclidean".
    return_vec_sil : bool, optional
        Should the simplified silhouette width of all single variables be 
        returned in addition to the simplified silhouette coefficient? The 
        default is False.

    Returns
    -------
    simplified_silhouette_coefficient : scalar
        The simplified silhouette coefficient.
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the silhouette width of all single variables 
        is additionally returned.
        
    """
    # Determine labels indicating different clusters
    unique_labels = np.unique(labels)
    # Determine number of observations
    N = X.shape[0]
    # Initiate matrix with cluster centroids
    mat_cetroid = np.zeros((N, unique_labels.shape[0]))
    # Calculate matrix with cluster centroids
    for l in range(unique_labels.shape[0]):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == unique_labels[l])[0]
        # Calculate centroid
        if metric == "euclidean":
            mat_cetroid[:, l] = np.mean(X[:, id_vox_l], 1)
        elif metric == "correlation":
            pca = PCA(n_components=1)
            pca.fit(X[:, id_vox_l])
            lat_comp = np.dot(X[:, id_vox_l], pca.components_.T).flatten()
            mat_cetroid[:, l] = lat_comp / np.linalg.norm(lat_comp)
        else:
            raise ValueError("Unknown metric: {0}".format(metric))
    # Initiate vec_s
    vec_s = np.array([])
    # For each cluster
    for l in range(unique_labels.shape[0]):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == unique_labels[l])[0]
        # Calculate data matrix of cluster l
        if id_vox_l.shape[0] == 1:
            vec_s = np.append(vec_s, 0)
            continue
        else:
            X_l = X[:, id_vox_l]
        # Within distances of cluster l       
        vec_a = distance.cdist(X_l.T, mat_cetroid[:, np.array([l])].T, metric = metric).flatten()
        if metric == "correlation":
            vec_a = 1 - np.abs((vec_a - 1)*(-1))
        
        #### Calculate vec_b
        # Initiate mat_b to later calculate column min from
        mat_b = np.zeros((unique_labels.shape[0] - 1, id_vox_l.shape[0]))
        # Initiate counter
        counter = 0
        for h in range(unique_labels.shape[0]):
            if h == l:
                continue
            # calculate vector with between distances
            tmp_vec_b = distance.cdist(X_l.T, mat_cetroid[:, np.array([h])].T, metric = metric).flatten()
            if metric == "correlation":
                mat_b[counter, :] = 1 - np.abs((tmp_vec_b - 1)*(-1))
            else:
                mat_b[counter, :] = tmp_vec_b
            # Update counter
            counter += 1
        # Calculate vector with between distance
        vec_b = np.min(mat_b, 0)
        # Calculate s
        vec_s_l = (vec_b - vec_a) / np.maximum(vec_a, vec_b)
        # Update vec_s
        vec_s = np.append(vec_s, vec_s_l)
    # Return simplified silhouette coefficient
    simplified_silhouette_coefficient = np.mean(vec_s)
    if return_vec_sil:
        return simplified_silhouette_coefficient, vec_s
    else:
        return simplified_silhouette_coefficient
    
def get_list_neighbors(matXYZ, diag_neighbor = False, print_progress = False):
    """
    Calculates list_neighors, which is needed for the calculation of the spatial
    adaptations of the silhouette coefficient. Note that it is recommended to 
    perform this calculation once for an input data set and to save the resulting 
    list_neighbors list to the hard drive. Whenever a parcellation generated
    based on the input data should be evaluated using the spatial silhouette 
    coefficient or the spatial simplified silhouette coefficient, list_neighbors 
    can be loaded from the hard drive instead of being calculated again, which
    is computationally expensive.

    Parameters
    ----------
    matXYZ : ndarray, shape(V, 3)
        Matrix of voxel coordinates. 
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors of 
        each voxel. If True, a maximum of 26 voxels belong to each voxel's 
        neighborhood. Default is False.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of the algorithm. Default is False.

    Returns
    -------
    list_neighbors : list, length (V,)
        A list of length V, where the j-th entry contains a numpy array with the 
        indices of the neighbor voxels of the j-th voxel.

    """
    # Number of voxels
    V = matXYZ.shape[0]
    # Initiate empty list which will be the output of our function
    list_neighbors = []
    # For each voxel determine its neighbors and save neighbor information
    # to list_neighbors
    for i in range(V):
        if print_progress:
            if i % 10000 == 0:
                print("Iteration " + str(i) + " of in total " + str(V) + " iterations to calculate list_neighbors.")
        if diag_neighbor:
            id_neighbors = np.where(np.max(abs(matXYZ[i,] - matXYZ),1) == 1)[0]
        else:
            id_neighbors = np.where(np.sum(abs(matXYZ[i,] - matXYZ), axis = 1) == 1)[0]
        list_neighbors.append(id_neighbors)
    # Return List with neighbors
    return list_neighbors


def silhouette_width_index_spatial(X, labels, list_neighbors = None, metric = "euclidean", 
                                   iter_max = 20000, return_vec_sil = False):
    """
    TODO How to generate list_neighbors

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    list_neighbors : TYPE
        DESCRIPTION.
    metric : TYPE, optional
        DESCRIPTION. The default is "euclidean".
    iter_max : TYPE, optional
        DESCRIPTION. The default is 20000.
    return_vec_sil : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    # Determine labels indicating different clusters
    unique_labels = np.unique(labels)
    # Initiate vec_s
    vec_s = np.array([])
    # Get list with information which clusters are neighbors
    list_neighbors_cluster = get_list_neighbors_cluster(labels, list_neighbors)
    # For each cluster
    for l in np.nditer(unique_labels):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == l)[0]
        # Get number of voxels in Cluster l
        n_voxels_l = id_vox_l.shape[0]
        # Calculate data matrix of cluster l
        if id_vox_l.shape[0] == 1:
            vec_s = np.append(vec_s, 0)
            continue
        else:
            X_l = X[:, id_vox_l]
        #### Divide voxels in cluster l in groups of maximum size = iter_max.
        # Get number of groups
        n_groups_l = int(np.ceil(np.array([n_voxels_l / iter_max])))
        # Initiate vec_a
        vec_a = np.zeros(n_voxels_l)
        # Compare each group with each group
        for i in range(n_groups_l):
            i_from = (i * iter_max)
            i_to = (np.min(((i + 1) * iter_max, n_voxels_l)))
            X_l_i = X_l[:, i_from:i_to]
            if metric == "correlation":
                tmp_mat = distance.squareform(distance.pdist(X_l_i.T, metric = metric))
                vec_scores = np.sum(1 - np.abs(1-tmp_mat), 0)
            else:
                vec_scores = np.sum(distance.squareform(distance.pdist(X_l_i.T, metric = metric)),0)
            vec_a[i_from:i_to] = vec_a[i_from:i_to] + vec_scores
            for j in range(i+1, n_groups_l):
                j_from = (j * iter_max)
                j_to = (np.min(((j + 1) * iter_max, n_voxels_l)))
                X_l_j = X_l[:, j_from:j_to]
                mat_dist_ij = distance.cdist(X_l_i.T, X_l_j.T, metric = metric)
                if metric == "correlation":
                    mat_dist_ij = 1 - np.abs(1-mat_dist_ij)
                vec_a[i_from:i_to] = vec_a[i_from:i_to] + np.sum(mat_dist_ij, 1)
                vec_a[j_from:j_to] = vec_a[j_from:j_to] + np.sum(mat_dist_ij, 0)
        # Calculate vector with within distances 
        vec_a = vec_a/(n_voxels_l - 1)
        #### Calculate vec_b
        # Determine ID of label l
        id_l = int(np.where(unique_labels == l)[0])
        # Find labels of neighbors of cluster l (note that neighbors of l do not include l)
        neighbors_l = list_neighbors_cluster[id_l]
        # Labels of clusters that are neighbors with cluster l
        # Initiate mat_b to later calculate column min from
        mat_b = np.zeros((neighbors_l.shape[0], n_voxels_l))
        # Initiate counter
        counter = 0
        for h in np.nditer(neighbors_l):
            # ID voxels belongig to cluster h
            id_vox_h = np.where(labels == h)[0]
            # Get number of voxels in Cluster h
            n_voxels_h = id_vox_h.shape[0]
            # Calculate data matrix of cluster h
            if n_voxels_h == 1:
                X_h = X[:, id_vox_h].reshape((X.shape[0], 1))
            else:
                X_h = X[:, id_vox_h] 
            # Get number of groups for cluster h
            n_groups_h = int(np.ceil(np.array([n_voxels_h / iter_max])))
            # Compare each group with each group
            for i in range(n_groups_l):
                i_from = (i * iter_max)
                i_to = (np.min(((i + 1) * iter_max, n_voxels_l)))
                X_l_i = X_l[:, i_from:i_to]
                for j in range(n_groups_h):
                    j_from = (j * iter_max)
                    j_to = (np.min(((j + 1) * iter_max, n_voxels_h)))
                    X_h_j = X_h[:, j_from:j_to]
                    if metric == "correlation":
                        tmp_mat = distance.cdist(X_l_i.T, X_h_j.T, metric = metric)
                        vec_scores = np.sum(1 - np.abs(1-tmp_mat), 1)
                    else:
                        vec_scores = np.sum(distance.cdist(X_l_i.T, X_h_j.T, metric = metric), 1)
                    mat_b[counter, i_from:i_to] = mat_b[counter, i_from:i_to] + vec_scores
            # Calculate matrix with between distances
            mat_b[counter, :] = mat_b[counter, :] / n_voxels_h
            # Update counter
            counter += 1
        # Calculate vector with between distance
        vec_b = np.min(mat_b, 0)
        # Calculate s
        vec_s_l = (vec_b - vec_a) / np.maximum(vec_a, vec_b)
        # Update vec_s
        vec_s = np.append(vec_s, vec_s_l)
    # Return silhouette index
    if return_vec_sil:
        return np.mean(vec_s), vec_s
    else:
        return np.mean(vec_s)
    
def simplified_silhouette_width_index_spatial(X, labels, list_neighbors, metric = "euclidean",
                           return_vec_sil = False):
    # Determine labels indicating different clusters
    unique_labels = np.unique(labels)
    # Determine number of observations
    N = X.shape[0]
    # Get list with information which clusters are neighbors
    list_neighbors_cluster = get_list_neighbors_cluster(labels, list_neighbors)
    # Initiate matrix with cluster centroids
    mat_cetroid = np.zeros((N, unique_labels.shape[0]))
    # Calculate matrix with cluster centroids
    for l in range(unique_labels.shape[0]):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == unique_labels[l])[0]
        # Calculate centroid
        if metric == "euclidean":
            mat_cetroid[:, l] = np.mean(X[:, id_vox_l], 1)
        elif metric == "correlation":
            pca = PCA(n_components=1)
            pca.fit(X[:, id_vox_l])
            lat_comp = np.dot(X[:, id_vox_l], pca.components_.T).flatten()
            mat_cetroid[:, l] = lat_comp / np.linalg.norm(lat_comp)
        else:
            raise ValueError("Unknown metric: {0}".format(metric))
    # Initiate vec_s
    vec_s = np.array([])
    # For each cluster
    for l in range(unique_labels.shape[0]):
        # ID voxels belongig to cluster l
        id_vox_l = np.where(labels == unique_labels[l])[0]
        # Calculate data matrix of cluster l
        if id_vox_l.shape[0] == 1:
            vec_s = np.append(vec_s, 0)
            continue
        else:
            X_l = X[:, id_vox_l]
        # Within distances of cluster l       
        vec_a = distance.cdist(X_l.T, mat_cetroid[:, np.array([l])].T, metric = metric).flatten()
        if metric == "correlation":
            vec_a = 1 - np.abs((vec_a - 1)*(-1))
        
        #### Calculate vec_b
        # Find labels of neighbors of cluster l (note that neighbors of l do not include l)
        neighbors_l = list_neighbors_cluster[l]
        # Labels of clusters that are neighbors with cluster l
        # Initiate mat_b to later calculate column min from
        mat_b = np.zeros((neighbors_l.shape[0], id_vox_l.shape[0]))
        # Initiate counter
        counter = 0
        for h in np.nditer(neighbors_l):
            # Get ID of neighbor h
            id_h = np.array([int(np.where(unique_labels == h)[0])])
            # calculate vector with between distances
            tmp_vec_b = distance.cdist(X_l.T, mat_cetroid[:, id_h].T, metric = metric).flatten()
            if metric == "correlation":
                mat_b[counter, :] = 1 - np.abs((tmp_vec_b - 1)*(-1))
            else:
                mat_b[counter, :] = tmp_vec_b
            # Update counter
            counter += 1
        # Calculate vector with between distance
        vec_b = np.min(mat_b, 0)
        # Calculate s
        vec_s_l = (vec_b - vec_a) / np.maximum(vec_a, vec_b)
        # Update vec_s
        vec_s = np.append(vec_s, vec_s_l)
    # Return simplified silhouette index
    if return_vec_sil:
        return np.mean(vec_s), vec_s
    else:
        return np.mean(vec_s)
    
def get_list_neighbors_cluster(labels, list_neighbors):
    """
    Generates list, where the j-th entry of that list contains the indices of 
    the neighbor clusters of the j-th cluster.
    """
    if labels.shape[0] != len(list_neighbors):
        raise ValueError("Dimensions of label vector and list with neighbor information do not match")
    # Vector with unique labels
    vec_uni_labels = np.unique(labels)
    # Number of clusters
    K = vec_uni_labels.shape[0]
    # Initiate output list
    list_neighbors_cluster = []
    # For each cluster
    for k in range(K):
        # Determine voxel IDs of k-th cluster 
        ids = np.where(labels == vec_uni_labels[k])[0]
        # Initiate neighbor array of k-th cluster
        uni_labels_neighbors = np.array([])
        # Get unique labels of neighbors
        for i in np.nditer(ids):
            uni_labels_neighbors = np.unique(np.append(uni_labels_neighbors, labels[list_neighbors[i].astype(int)]))
        # Remove label of k-th cluster from that list
        list_neighbors_cluster.append(np.delete(uni_labels_neighbors, obj = np.where(uni_labels_neighbors==vec_uni_labels[k])[0]).astype(int))
    # Return list
    return list_neighbors_cluster
    
###############################################################################
if __name__ == "__main__" and __name__ != "__main__":
    import sys
    sys.path.append("C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019/Funktionen")
    import Plot_ensemble_brain_image as bi
    import spatial_hierarchical_variable_clusteringV2 as vc
    import sklearn.metrics as metrics
    import scipy.io
    import os
    import csv
    from sklearn.preprocessing import StandardScaler
    # Test Silhouette function
    # 1. arrayA
    os.chdir("C:/Users/admin/Documents/Promotion_Teil_2_MRI/Python/Python_Code/Python_Code")
    
    with open("matrixA.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        matA = []
        for row in readCSV:
            matA.append(row)
    arrayA = np.array(matA)
    arrayA = arrayA[1:arrayA.shape[0],:]
    arrayA = arrayA[:,1:arrayA.shape[1]]
    arrayA = arrayA.astype(int)
    X = arrayA.copy()
    
    matXYZ = np.ones((arrayA.shape[1], 3), dtype = int)
    
    Z = vc.hclustvar_spatial(arrayA, matXYZ, method = 'ward', metric = 'euclidean', 
                          standardize = False)
    labels = vc.get_cluster(Z, X.shape[1], 5)    
        
    silhouette_width_index(X, labels, metric = "correlation", iter_max = 10)    
        
    metrics.silhouette_score(X.T, labels, metric = "correlation")    
        
        
        
    ###############################################################################
    # 2. Further flexible tests 
    n_voxel = 400
    N = 100
    labels =  np.random.randint(1,11,n_voxel)   
    
    X = np.random.normal(size = (N, n_voxel))
        
    silhouette_width_index(X, labels, metric = "correlation")    
        
    metrics.silhouette_score(X.T, labels, metric = "correlation")  
    
    
    ###############################################################################  
    # 3. Test on real data
    # Read in data set
    standardize = True
    mat = scipy.io.loadmat('C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019/Daten/FZJ1000_smoothdat.mat')    
    matFZJ693 = mat["GLMFlags"].item(0)[2]
    matFZJ693 = np.delete(matFZJ693, np.array([126961, 126962, 126992, 126993, 18602], dtype = int), 1)
    matFZJ693 = np.delete(matFZJ693, np.array([323,324], dtype = int), 0)
    # Standardize data
    if standardize:
        matFZJ693 = StandardScaler().fit_transform(matFZJ693)
        matFZJ693 = matFZJ693 / np.sqrt(matFZJ693.shape[0] / (matFZJ693.shape[0] - 1))
    
    X = matFZJ693[:200,:]
    dir_from = "C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019"
    os.chdir(dir_from)
    # Get labels
    n_cluster = 83
    vec_cluster = bi.determine_clustering(n_cluster, ensemble_method = "subsample",
                                           cluster_algo = "lacomp_", sampling = "subsampling")
    silhouette_width_index(X, vec_cluster, iter_max = 30000)

    ###########################################################################
    # 4. Test ensemble silhouette coefficient
    X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])   
    labels = [1,1,2,2,3,3,4,4]
    silhouette_width_index(X[0:4,:], labels, metric = "jaccard", iter_max = 20000)
    
    X1 = np.array([[1,1,2,2], [1,2,2,2], [1,1,1,2]])
    labels1 = [1,1,2,2]
    silhouette_width_index(X1, labels1, metric = "jaccard", iter_max = 20000)
    # Checked by hand. Same results :-) 2*(0.6+1/3)/4
