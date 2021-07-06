# -*- coding: utf-8 -*-
"""
This module includes implementations of the popular silhouette coefficient 
(silhouette_coefficient()) for large data sets (avoiding to run into a memory 
error) as introduced by Rousseeuw (1987), the computationally cheaper 
simplified silhouette coefficient (simplified_silhouette_coefficient()) as 
introduced by Vendramin et al. (2010), as well as spatial adaptations thereof
(silhouette_coefficient_spatial(), or simplified_silhouette_coefficient_spatial(), 
respectively), which are developed to evaluate partitions of images with 
spatially contiguous clusters (see Tietz et al., 2021).

References
----------
Rousseeuw PJ (1987) Silhouettes: a graphical aid to the interpretation and 
        validation of cluster analysis. Journal of computational and applied 
        mathematics 20:53-65
Vendramin L, Campello RJGB, Hruschka ER (2010) Relative clustering validity 
        criteria: A comparative overview. Statistical analysis and data mining: 
        the ASA data science journal 3(4):209-235
Tietz et al. (2021) (Publication in progress)

"""

import numpy as np
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA

def silhouette_coefficient(X: np.ndarray, labels: np.ndarray, metric = "euclidean", iter_max = 20000,
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
    SC : scalar
        The silhouette coefficient.
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the silhouette width of all single variables 
        is additionally returned.
    
    Examples
    --------
    Random input data with 100 variables and a partition assigning each variable
    to one of in total four clusters:
        
    >>> X = np.random.normal(size = (50, 100))
    >>> labels = np.random.randint(1, 5, 100)
    >>> silhouette_coefficient(X, labels, metric = "euclidean")   
    -0.0171145
    
    References
    ----------
    Rousseeuw PJ (1987) Silhouettes: a graphical aid to the interpretation and 
        validation of cluster analysis. Journal of computational and applied 
        mathematics 20:53-65
        
    """

    # get_silhouette_width_of_single_variable
    # get_average_silhouette_width_of_cluster


    unique_labels = np.unique(labels)
    silhouette_width_of_single_voxel = np.array([])
    for cluster in np.nditer(unique_labels):
        voxelsBelongingToCluster = np.where(labels == cluster)[0]
        numberVoxelsInCluster = voxelsBelongingToCluster.shape[0]
        if numberVoxelsInCluster == 1:
            silhouette_width_of_single_voxel = np.append(silhouette_width_of_single_voxel, 0)
            continue
        else:
            dataMatrixOfCluster = X[:, voxelsBelongingToCluster]
        #### Calculate a. Therefore:
        #### Divide voxels in cluster into groups of maximum size equal to iter_max.
        numberOfGroups = int(np.ceil(np.array([numberVoxelsInCluster / iter_max])))
        meanWithinClusterDistances = np.zeros(numberVoxelsInCluster) # a
        # Compare each group with each group
        for group1 in range(numberOfGroups):
            firstVoxelFromGroup1 = (group1 * iter_max)
            lastVoxelFromGroup1 = (np.min(((group1 + 1) * iter_max, numberVoxelsInCluster)))
            dataMatrixOfGroup1 = dataMatrixOfCluster[:, firstVoxelFromGroup1:lastVoxelFromGroup1]
            meanWithinGroup1Distances = getMeanWithinClusterDistances(dataMatrixOfGroup1, metric)
            meanWithinClusterDistances[firstVoxelFromGroup1:lastVoxelFromGroup1] += meanWithinGroup1Distances
            for group2 in range(group1+1, numberOfGroups):
                firstVoxelFromGroup2 = (group2 * iter_max)
                lastVoxelFromGroup2 = (np.min(((group2 + 1) * iter_max, numberVoxelsInCluster)))
                dataMatrixOfGroup2 = dataMatrixOfCluster[:, firstVoxelFromGroup2:lastVoxelFromGroup2]
                meanBetweenGroupDistances = getMeanBetweenClusterDistances(dataMatrixOfGroup1, dataMatrixOfGroup2, metric)
                meanWithinClusterDistances[firstVoxelFromGroup1:lastVoxelFromGroup1] += np.sum(meanBetweenGroupDistances, 1)
                meanWithinClusterDistances[firstVoxelFromGroup2:lastVoxelFromGroup2] += np.sum(meanBetweenGroupDistances, 0) 
        meanWithinClusterDistances = meanWithinClusterDistances/(numberVoxelsInCluster - 1)
        #### Calculate b
        # Initiate mat_b to later calculate column min from
        mat_b = np.zeros((unique_labels.shape[0] - 1, numberVoxelsInCluster))
        # Initiate counter
        counter = 0
        for h in np.nditer(unique_labels):
            if h == cluster:
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
            for group1 in range(numberOfGroups):
                firstVoxelFromGroup1 = (group1 * iter_max)
                lastVoxelFromGroup1 = (np.min(((group1 + 1) * iter_max, numberVoxelsInCluster)))
                dataMatrixOfGroup1 = dataMatrixOfCluster[:, firstVoxelFromGroup1:lastVoxelFromGroup1]
                for group2 in range(n_groups_h):
                    firstVoxelFromGroup2 = (group2 * iter_max)
                    lastVoxelFromGroup2 = (np.min(((group2 + 1) * iter_max, n_voxels_h)))
                    X_h_j = X_h[:, firstVoxelFromGroup2:lastVoxelFromGroup2]
                    if metric == "correlation":
                        tmp_mat = distance.cdist(dataMatrixOfGroup1.T, X_h_j.T, metric = metric)
                        vec_scores = np.sum(1 - np.abs(1-tmp_mat), 1)
                    else:
                        vec_scores = np.sum(distance.cdist(dataMatrixOfGroup1.T, X_h_j.T, metric = metric), 1)
                    mat_b[counter, firstVoxelFromGroup1:lastVoxelFromGroup1] = mat_b[counter, firstVoxelFromGroup1:lastVoxelFromGroup1] + vec_scores
            # Calculate matrix with between distances
            mat_b[counter, :] = mat_b[counter, :] / n_voxels_h
            # Update counter
            counter += 1
        # Calculate vector with between distance
        vec_b = np.min(mat_b, 0)
        # Calculate s
        vec_s_l = (vec_b - meanWithinClusterDistances) / np.maximum(meanWithinClusterDistances, vec_b)
        silhouette_width_of_single_voxel = np.append(silhouette_width_of_single_voxel, vec_s_l)
    # Return silhouette index
    SC = np.mean(silhouette_width_of_single_voxel)
    if return_vec_sil:
        return SC, silhouette_width_of_single_voxel
    else:
        return SC

def getMeanWithinClusterDistances(dataMatrixOfCluster: np.ndarray, metric: str) -> np.ndarray:
    """  
    Calculates for each voxel from the cluster the mean distance to all other
    voxels from the cluster. 

    Parameters
    ----------
    dataMatrixOfCluster : ndarray, shape(N, V_cluster)
        Input matrix including, e.g., grey matter volume values, where N is the 
        number of subjects and V_cluster is the number of voxels in that cluster. 
    metric : str
        The distance metric between two voxels. It is recommended to either
        employ 'euclidean' (Euclidean distance) or 'correlation' (1-abs(corr)).
        However, this function technically allows all distance metrics as 
        implemented in ?scipy.spatial.distance.pdist. In the latter case, it is 
        up to the user to ensure the validity of the chosen metric. 
    """
    if metric == "correlation":
        tmp_mat = distance.squareform(distance.pdist(dataMatrixOfCluster.T, metric = metric))
        return np.sum(1 - np.abs(1-tmp_mat), 0)
    else:
        return np.sum(distance.squareform(distance.pdist(dataMatrixOfCluster.T, metric = metric)),0)


def getMeanBetweenClusterDistances(dataMatrixOfCluster1: np.ndarray, dataMatrixOfCluster2: np.ndarray, metric: str) -> np.ndarray:
    """  
    Calculates for each voxel from the first cluster the mean distance to all 
    voxels from the second cluster. 

    Parameters
    ----------
    dataMatrixOfCluster1 : ndarray, shape(N, V_cluster1)
        Input matrix of the first cluster, where N is the 
        number of subjects and V_cluster1 is the number of voxels in that cluster.
    dataMatrixOfCluster2 : ndarray, shape(N, V_cluster2)
        Input matrix of the second cluster, where N is the 
        number of subjects and V_cluster2 is the number of voxels in that cluster.
    metric : str
        The distance metric between two variables. It is recommended to either
        employ 'euclidean' (Euclidean distance) or 'correlation' (1-abs(corr)).
        However, this function technically allows all distance metrics as 
        implemented in ?scipy.spatial.distance.pdist. In the latter case, it is 
        up to the user to ensure the validity of the chosen metric. 
    """

    matDist = distance.cdist(dataMatrixOfCluster1.T, dataMatrixOfCluster2.T, metric = metric)
    if metric == "correlation":
        matDist = 1 - np.abs(1-matDist)
    return matDist

def simplified_silhouette_coefficient(X, labels, metric = "euclidean",
                                      return_vec_sil = False):
    """
    Calculates simplified silhouette coefficient (SSC). For large data sets, pairwise 
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
    SSC : scalar
        The simplified silhouette coefficient (SSC).
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the SSC of all single variables is 
        additionally returned.
        
    Notes
    -----
    The SSC using Euclidean or correlation distance is introduced by 
    Vendramin et al. (2010) or Tietz et al. (2021), respectively. 
     
    Examples
    --------
    Random input data with 100 variables and a partition assigning each variable
    to one of in total four clusters:
        
    >>> X = np.random.normal(size = (50, 100))
    >>> labels = np.random.randint(1, 5, 100)
    >>> simplified_silhouette_coefficient(X, labels, metric = "euclidean")   
    0.01753568
    
    References
    ----------
    Vendramin L, Campello RJGB, Hruschka ER (2010) Relative clustering validity 
        criteria: A comparative overview. Statistical analysis and data mining: 
        the ASA data science journal 3(4):209-235
    Tietz et al. (2021) (Publication in progress)
        
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
    SSC = np.mean(vec_s)
    if return_vec_sil:
        return SSC, vec_s
    else:
        return SSC
    
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
    return list_neighbors


def silhouette_coefficient_spatial(X, labels, list_neighbors, metric = "euclidean", 
                                   iter_max = 20000, return_vec_sil = False):
    """
    Calculates the spatial adaptation of the silhouette coefficient. 

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
    list_neighbors : list, length (V,)
        List including the neighbor information of each variable, as returned
        by the get_list_neighbors() function. The j-th entry must contain a 
        numpy array with the indices of the neighbors of variable j, j=1,...,V. 
        Note that it is recommended to calculate list_neighbors once for an 
        input data set and to save it to the hard drive. Whenever a parcellation 
        generated based on the input data should be evaluated using 
        silhouette_coefficient_spatial(), list_neighbors can be loaded from the 
        hard drive instead of being re-calculated, which is computationally 
        expensive.
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
    SC_spatial : scalar
        The spatial silhouette coefficient.
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the spatial silhouette coefficient of all 
        single variables is additionally returned.
     
    Examples
    --------
    Random input data with 192 spatial variables on a 8x6x4 grid and a 
    partition assigning each variable to one of in total four spatially
    contiguous clusters:
        
    >>> X = np.random.normal(size = (50, 192))
    >>> matXYZ = np.argwhere(np.zeros((8, 6, 4)) == 0)
    >>> labels = np.repeat(np.array([1,2,3,4]), 2*6*4)
    >>> list_neighbors = get_list_neighbors(matXYZ) # Best to save list_neighbors
                                                    # to hard drive
    >>> silhouette_coefficient_spatial(X, labels, list_neighbors, metric = "euclidean")   
    -0.00466234
    
    References
    ----------
    Tietz et al. (2021) (Publication in progress.)

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
    # Return spatial silhouette index
    SC_spatial = np.mean(vec_s)
    if return_vec_sil:
        return SC_spatial, vec_s
    else:
        return SC_spatial
    
def simplified_silhouette_coefficient_spatial(X, labels, list_neighbors, metric = "euclidean",
                           return_vec_sil = False):
    """
    Calculates the spatial adaptation of the simplified silhouette coefficient. 

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
    list_neighbors : list, length (V,)
        List including the neighbor information of each variable, as returned
        by the get_list_neighbors() function. The j-th entry must contain a 
        numpy array with the indices of the neighbors of variable j, j=1,...,V. 
        Note that it is recommended to calculate list_neighbors once for an 
        input data set and to save it to the hard drive. Whenever a parcellation 
        generated based on the input data should be evaluated using 
        silhouette_coefficient_spatial(), list_neighbors can be loaded from the 
        hard drive instead of being re-calculated, which is computationally 
        expensive.
    metric : str, optional
        The distance metric between two variables. One of 'euclidean' 
        (Euclidean distance) or 'correlation' (1-abs(corr)). The default is 
        "euclidean".
    iter_max : int, optional
        In order to avoid a memory error, the pairwise variable distance 
        calculation is partitioned, if at least one of the two clusters includes 
        more than iter_max variables. The default is 20000.
    return_vec_sil : bool, optional
        Should the spatial simplified silhouette width of all single variables 
        be returned in addition to the spatial simplified silhouette 
        coefficient? The default is False.

    Returns
    -------
    SSC_spatial : scalar
        The spatial simplified silhouette coefficient.
    vec_s : ndarray, shape (V,)
        If return_vec_sil = True, the spatial simplified silhouette coefficient 
        of all single variables is additionally returned.
     
    Examples
    --------
    Random input data with 192 spatial variables on a 8x6x4 grid and a 
    partition assigning each variable to one of in total four spatially
    contiguous clusters:
        
    >>> X = np.random.normal(size = (50, 192))
    >>> matXYZ = np.argwhere(np.zeros((8, 6, 4)) == 0)
    >>> labels = np.repeat(np.array([1,2,3,4]), 2*6*4)
    >>> list_neighbors = get_list_neighbors(matXYZ) # Best to save list_neighbors
                                                    # to hard drive
    >>> simplified_silhouette_coefficient_spatial(X, labels, list_neighbors, metric = "euclidean")   
    0.01231799
    
    References
    ----------
    Vendramin L, Campello RJGB, Hruschka ER (2010) Relative clustering validity 
        criteria: A comparative overview. Statistical analysis and data mining: 
        the ASA data science journal 3(4):209-235
    Tietz et al. (2021) (Publication in progress.)

    """
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
    SSC_spatial = np.mean(vec_s)
    if return_vec_sil:
        return SSC_spatial, vec_s
    else:
        return SSC_spatial
    
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
    
