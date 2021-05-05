# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:43:29 2018

@author: Tobias Tietz
"""

import warnings
import numpy as np
import scipy
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

_LINKAGE_METHODS = ('single', 'complete', 'average', 'centroid', 'median', 'ward')
_DISTANCE_METHODS = ('squared_correlation', 'correlation', 'euclidean', 'spartacus')

###############################################################################
##### SHAC methods including SPARTACUS
###############################################################################

def shac(X, matXYZ, method='ward', metric='spartacus', diag_neighbor = False, 
         standardize = True, print_progress = False):
    """    
    Perform spatial hierarchical agglomerative clustering (SHAC). Note that
    the objects to be clustered are the variables and not the subjects.
    
    Parameters
    ----------
    X : ndarray, shape(N, V)
        Input matrix including, e.g., grey matter volume values, where N is 
        the number of subjects and V is the number of variables, e.g. voxels.
    matXYZ : ndarray, shape(V, 3)
        Matrix of variable coordinates. Rows include variable locations sorted 
        as in the column of X, i.e. the coordinates stored in the j-th row 
        of matXYZ correspond to the variable stored in the j-th column of X.
    method : str, optional
        The linkage method. One of 'single', 'complete, 'average', 'centroid',
        'median' or 'ward'. Only relevant, if metric is not 'spartacus'. 
        Default is 'ward'.
    metric : str, optional
        The linkage metric. One of 'euclidean', 'squared_correlation',
        'correlation' or 'spartacus'. If equal to 'spartacus', the SPARTACUS
        method is performed, ignoring the selection of method. 
        Default is 'spartacus'.
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors of 
        each voxel. If True, a maximum of 26 voxels belong to each voxel's 
        neighborhood. Default is False.
    standardize : bool, optional
        Should the columns of the input matrix be standardized? 
        Default is True.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of initializing the sparse distance matrix and with every 
        ten thousandth iteration of the main algorithm. Default is False.
        
    Returns
    -------
    Z : ndarray, shape (V - a, 4)
        Computed linkage matrix, where 'a' is the number of contiguous regions
        of variables included in the input data.
        
    """   
    if metric not in _DISTANCE_METHODS:
        raise ValueError("Invalid metric: {0}".format(metric))
    if metric != "spartacus":    
        if method not in _LINKAGE_METHODS:
            raise ValueError("Invalid method: {0}".format(method))
    if (method in ["median", "ward", "centroid"]) and (metric != 'euclidean'):
        raise ValueError("Method {0} is only defined, if euclidean pairwise "
                         "metric is used.".format(method))
    if X.ndim == 2:
        if metric not in _DISTANCE_METHODS:
            raise ValueError("Unknown distance metric.")
        X = hierarchy._convert_to_double(np.asarray(X, order='c'))
        if X.shape[0] == X.shape[1] and np.allclose(np.diag(X), 0):
            if np.all(X >= 0) and np.allclose(X, X.T):
                warnings.warn('The symmetric non-negative hollow observation '
                              'matrix looks suspiciously like an uncondensed '
                              'distance matrix.')
        if X.shape[0] > X.shape[1]:
            warnings.warn('For the two dimensional intensity matrix the number '
                          'of colums (i.e. the number of voxels) is smaller than '
                          'the number of rows (i.e. the number of pictures). '
                          'You may want to consider the transpose of X instead.')
    else:
        raise ValueError("'X' must be 2 dimensional.")
    if not np.all(np.isfinite(X)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")
    # Checks for matXYZ 
    if matXYZ.ndim != 2:
        raise ValueError("Neighbor matrix 'matXYZ' must be 2 dimensional.")
    if matXYZ.shape[1] != 3:
        raise ValueError("Neighbor matrix 'matXYZ' must have three spatial coordinates.")
    if not (np.round(matXYZ)==matXYZ).all():
        raise ValueError("All entries in matXYZ must be integers.")
    # Standardize the columns of the input matrix?
    if standardize:
        X = StandardScaler().fit_transform(X)
        X = X / np.sqrt(X.shape[0] / (X.shape[0] - 1))
    # Perform either SHAC or SPARTACUS
    if metric in ("squared_correlation", "correlation", "euclidean"):
        Z = shac_linkage(X, matXYZ, method, metric, diag_neighbor, print_progress)
    else:
        Z = shac_spartacus(X, matXYZ, diag_neighbor, print_progress)
    # Return linkage matrix
    return Z

def shac_spartacus(X, matXYZ, diag_neighbor = False, print_progress = False):
    """
    SPARTACUS method.
    
    Parameters
    ----------
    X : ndarray, shape(N, V)
        Input matrix including, e.g., grey matter volume values, where N is 
        the number of subjects and V is the number of variables, e.g. voxels.
    matXYZ : ndarray, shape(V, 3)
        Matrix of variable coordinates. Rows include variable locations sorted 
        as in the column of X, i.e. the coordinates stored in the j-th row 
        of matXYZ correspond to the variable stored in the j-th column of X.
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors of 
        each voxel. If True, a maximum of 26 voxels belong to each voxel's 
        neighborhood. Default is False.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of initializing the sparse distance matrix and with every 
        ten thousandth iteration of the main algorithm. Default is False.
    
    Returns
    -------
    Z : ndarray, shape (V - a, 4)
        Computed linkage matrix, where 'a' is the number of contiguous regions
        of variables included in the input data.
        
    """    
    V = matXYZ.shape[0]
    X_cluster = []
    vec_lambda = np.var(X, axis = 0, ddof = 1)
    for i in range(V):
        X_cluster.append(np.array([i]))
    vec_dist = np.array([])
    vec_index1 = np.array([], dtype=np.int)
    vec_index2 = np.array([], dtype=np.int)
    # Initialization of sparse distance matrix calculating only distances 
    # between neighbor-variables
    for i in np.nditer(np.arange(V)):
        if print_progress:
            if (i % 10000) == 0:
                print("Initial spartacus: Iteration " + str(i))
        if diag_neighbor:
            id_neighbor = np.where(np.max(abs(matXYZ[i,] - matXYZ[(i+1):,]),1) <= 1)[0]+i+1
        else:
            id_neighbor = np.where(np.sum(abs(matXYZ[i,] - matXYZ[(i+1):,]), axis = 1) <= 1)[0]+i+1
        vec_index1 = np.append(vec_index1, np.array([i]*len(id_neighbor), dtype=np.int))
        vec_index2 = np.append(vec_index2, id_neighbor)
        if len(id_neighbor) > 0:
            for j in np.nditer(id_neighbor):
                vec_dist = np.append(vec_dist, get_dist_spartacus(X, i, j, 
                                                vec_lambda[i], vec_lambda[j]))
    # Main SPARTACUS algorithm
    Z_arr = np.empty((V - 1, 4))
    size = np.ones(V, dtype=np.int)  # Sizes of clusters.
    for k in range(V - 1):
        if print_progress:
            if (k % 10000) == 0:
                print("Main spartacus: Iteration " + str(k))
        if len(vec_dist) == 0:
            for l in range(k, V - 1):
                Z_arr = np.delete(Z_arr, k, 0)
            print(V - k, "contiguous regions in data set.")
            break
        index = np.argmin(vec_dist)
        current_min = vec_dist[index]
        x = vec_index1[index]
        y = vec_index2[index]
        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]
        # Record the new node.
        Z_arr[k, 0] = x
        Z_arr[k, 1] = y
        Z_arr[k, 2] = current_min
        Z_arr[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster
        # Update X_cluster and then vec_lambda
        X_cluster[y] = np.concatenate((X_cluster[y], X_cluster[x]))
        pca = PCA(n_components=1)
        pca.fit(X[:, X_cluster[y]])
        vec_lambda[y] = pca.explained_variance_
        if k == V - 2:
            break
        # Find neighbors of x and y:
        xy_neighbor = np.concatenate((vec_index2[np.where(vec_index1 == x)[0]], 
                                      vec_index2[np.where(vec_index1 == y)[0]],
                                      vec_index1[np.where(vec_index2 == x)[0]],
                                      vec_index1[np.where(vec_index2 == y)[0]]))    
        # Remove x and y from this list:
        xy_neighbor = xy_neighbor[xy_neighbor != x] 
        xy_neighbor = xy_neighbor[xy_neighbor != y]   
        # Get unique neighbors
        xy_neighbor = np.unique(xy_neighbor)
        # Calculate distance of cluster y to its neighbors
        tmp_vec_index1 = np.array([], dtype = int)
        tmp_vec_index2 = np.array([], dtype = int)
        tmp_vec_dist = np.array([], dtype = int)
        if len(xy_neighbor) > 0:
            for i in np.nditer(xy_neighbor):
                dst = get_dist_spartacus(X, X_cluster[i], X_cluster[y], 
                                      vec_lambda[i], vec_lambda[y])
                if i < y:
                    tmp_vec_index1 = np.append(tmp_vec_index1, i)
                    tmp_vec_index2 = np.append(tmp_vec_index2, y)
                else:
                    tmp_vec_index1 = np.append(tmp_vec_index1, y)
                    tmp_vec_index2 = np.append(tmp_vec_index2, i)
                tmp_vec_dist = np.append(tmp_vec_dist, dst)
        # Remove all distances of x and y to its respective neighbors 
        id1 = np.unique(np.concatenate((np.where(vec_index1 == x)[0],
                                         np.where(vec_index1 == y)[0],
                                         np.where(vec_index2 == x)[0],
                                         np.where(vec_index2 == y)[0])))
        vec_index1 = np.delete(vec_index1, id1)
        vec_index2 = np.delete(vec_index2, id1)
        vec_dist = np.delete(vec_dist, id1)
        # Add new distances
        vec_index1 = np.append(vec_index1, tmp_vec_index1)
        vec_index2 = np.append(vec_index2, tmp_vec_index2)
        vec_dist = np.append(vec_dist, tmp_vec_dist)  
    # Return linkage matrix
    Z = get_label(Z_arr, V)
    return Z

def get_dist_spartacus(X, clus_i, clus_j, lambda_i, lambda_j):
    """
    Calculates the SPARTACUS distance between two (adjacent) clusters, i.e.
    calculates eigenvalue of cluster i plus eigenvalue of cluster j minus 
    the eigenvalue of the combined cluster.
    """
    if type(clus_i) != np.ndarray or type(clus_j) != np.ndarray:
        raise ValueError("clus_i and clus_j must be of type numpy array.")
    pca = PCA(n_components=1)
    pca.fit(X[:, np.append(clus_i, clus_j)])
    lambda_ij = pca.explained_variance_
    return lambda_i + lambda_j - lambda_ij

def shac_linkage(X, matXYZ, method, metric, diag_neighbor = False, 
                 print_progress = False):
    """
    SHAC using popular linkage methods.
    
    Parameters
    ----------
     X : ndarray, shape(N, V)
        Input matrix including, e.g., grey matter volume values, where N is 
        the number of subjects and V is the number of variables, e.g. voxels.
    matXYZ : ndarray, shape(V, 3)
        Matrix of variable coordinates. Rows include variable locations sorted 
        as in the column of X, i.e. the coordinates stored in the j-th row 
        of matXYZ correspond to the variable stored in the j-th column of X.
    method : str
        The linkage method. One of 'single', 'complete, 'average', 'centroid',
        'median' or 'ward'.
    metric : str
        The linkage metric. One of 'euclidean', 'squared_correlation' or
        'correlation'.    
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors of 
        each voxel. If True, a maximum of 26 voxels belong to each voxel's 
        neighborhood. Default is False.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of initializing the sparse distance matrix and with every 
        ten thousandth iteration of the main algorithm. Default is False.
        
    Returns
    -------
    Z : ndarray, shape (V - a, 4)
        Computed linkage matrix, where 'a' is the number of contiguous regions
        of variables included in the input data.
        
    """
    if (method in ["median", "ward", "centroid"]) and (metric != 'euclidean'):
        raise ValueError("Method {0} is only defined, if euclidean pairwise "
                         "metric is used.".format(method))
    V = matXYZ.shape[0]
    if method in ["ward", "centroid"]:
        X_cluster = X.copy()
    elif method in ["single", "complete", "average"]:
        X_cluster = []
        for i in range(V):
            X_cluster.append([i])
    elif method == "median":
        X_cluster = []
        for i in range(V):
            X_cluster.append(X[:, i])
    else:
        ValueError("Method '{0}' not implemented so far.".format(method))
    # Initialization of sparse distance matrix calculating only distances 
    # between neighbor-variables
    vec_dist = np.array([])
    vec_index1 = np.array([], dtype=np.int)
    vec_index2 = np.array([], dtype=np.int)
    for i in range(V):
        if print_progress:
            if (i % 10000) == 0:
                print("Initial linkage: Iteration " + str(i))
        if diag_neighbor:
            id_neighbor = np.where(np.max(abs(matXYZ[i,] - matXYZ[(i+1):,]),1) <= 1)[0]+i+1
        else:
            id_neighbor = np.where(np.sum(abs(matXYZ[i,] - matXYZ[(i+1):,]), axis = 1) <= 1)[0]+i+1
        vec_index1 = np.append(vec_index1, np.array([i]*len(id_neighbor), dtype=np.int))
        vec_index2 = np.append(vec_index2, id_neighbor)
        vec_dist = np.append(vec_dist, get_dist(X, i, id_neighbor, metric))
    if method == "ward":
        vec_dist = vec_dist**2/2
    # Main SHAC algorithm
    Z_arr = np.empty((V - 1, 4))
    size = np.ones(V, dtype=np.int)  # Sizes of clusters.
    for k in range(V - 1):
        if print_progress:
            if (k % 10000) == 0:
                print("Main linkage: Iteration " + str(k))
        if len(vec_dist) == 0:
            for l in range(k, V - 1):
                Z_arr = np.delete(Z_arr, k, 0)
            print(V - k, "contiguous regions in data set.")
            break
        index = np.argmin(vec_dist)
        current_min = vec_dist[index]
        x = vec_index1[index]
        y = vec_index2[index]
        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]
        # Record the new node.
        Z_arr[k, 0] = x
        Z_arr[k, 1] = y
        Z_arr[k, 2] = current_min
        Z_arr[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster
        # Break, if all variables are in the same cluster
        if k == V - 2:
            break
        # Find neighbors of x and y
        xy_neighbor = np.concatenate((vec_index2[np.where(vec_index1 == x)[0]], 
                                      vec_index2[np.where(vec_index1 == y)[0]],
                                      vec_index1[np.where(vec_index2 == x)[0]],
                                      vec_index1[np.where(vec_index2 == y)[0]]))    
        xy_neighbor = xy_neighbor[xy_neighbor != x]   # Remove x and y from list xy_neighbor
        xy_neighbor = xy_neighbor[xy_neighbor != y]  
        xy_neighbor = np.unique(xy_neighbor)   # Calculate unique neighbors   
        # Calculated distance of new cluster to its neighbors
        if method in ["ward", "centroid"]:
            X_cluster[:,y] = X_cluster[:,x] + X_cluster[:,y]
        elif method == "median":
            X_cluster[y] = 1/2 * (X_cluster[x] + X_cluster[y])
        tmp_vec_index1 = np.array([], dtype = int)
        tmp_vec_index2 = np.array([], dtype = int)
        tmp_vec_dist = np.array([], dtype = int)
        if len(xy_neighbor) > 0:
            for i in np.nditer(xy_neighbor):
                if method in ["ward", "centroid"]:
                    dst = get_cluster_dist_mean(X_cluster, size, i, y, method)
                elif method == "median":
                    dst = get_cluster_dist_median(X_cluster, i, y)
                else: 
                    # Find out, whether i is neighbor to x and/or y
                    if i < x:
                        ix = np.where((vec_index1==i) * (vec_index2 == x))[0]
                        iy = np.where((vec_index1==i) * (vec_index2 == y))[0]
                    elif i > y:
                        ix = np.where((vec_index1==x) * (vec_index2 == i))[0]
                        iy = np.where((vec_index1==y) * (vec_index2 == i))[0]
                    else:
                        ix = np.where((vec_index1==x) * (vec_index2 == i))[0]
                        iy = np.where((vec_index1==i) * (vec_index2 == y))[0]
                    # Calculate distance of i to x and i to y
                    if len(ix) == 1:
                        dist_ix = vec_dist[ix]
                    else:
                        dist_ix = get_cluster_dist_linkage(X, X_cluster, i, x, method, metric)
                    if len(iy) == 1:
                        dist_iy = vec_dist[iy]
                    else:
                        dist_iy = get_cluster_dist_linkage(X, X_cluster, i, y, method, metric)
                    # Use dist_ix and dist_iy to calculate distance of i to the 
                    # newly merged cluster. 
                    # single linkage
                    if method == "single":
                        dst = min(dist_ix, dist_iy)
                    # complete linkage
                    elif method == "complete":
                        dst = max(dist_ix, dist_iy)
                    # average linkage
                    else:
                        N_x = len(X_cluster[x])
                        N_y = len(X_cluster[y])
                        dst = N_x/(N_x+N_y)*dist_ix + N_y/(N_x+N_y)*dist_iy               
                if i < y:
                    tmp_vec_index1 = np.append(tmp_vec_index1, i)
                    tmp_vec_index2 = np.append(tmp_vec_index2, y)
                else:
                    tmp_vec_index1 = np.append(tmp_vec_index1, y)
                    tmp_vec_index2 = np.append(tmp_vec_index2, i)
                tmp_vec_dist = np.append(tmp_vec_dist, dst)
        # Remove all distances of x and y to their neighbors from vec_dist
        id1 = np.unique(np.concatenate((np.where(vec_index1 == x)[0],
                                         np.where(vec_index1 == y)[0],
                                         np.where(vec_index2 == x)[0],
                                         np.where(vec_index2 == y)[0])))
        vec_index1 = np.delete(vec_index1, id1)
        vec_index2 = np.delete(vec_index2, id1)
        vec_dist = np.delete(vec_dist, id1)
        # Add new distances to vec_dist 
        vec_index1 = np.append(vec_index1, tmp_vec_index1)
        vec_index2 = np.append(vec_index2, tmp_vec_index2)
        vec_dist = np.append(vec_dist, tmp_vec_dist)
        # Update X_cluster
        if method not in ["ward", "centroid", "median"]:
            X_cluster[y].extend(X_cluster[x])
    # Return linkage matrix
    Z = get_label(Z_arr, V)
    return Z

def get_dist(X, i, id_neighbor, metric):
    """
    Calculates distance between variable i and each variable in id_neighbor
    """
    dst = np.array([])
    if len(id_neighbor) > 0:
        for j in np.nditer(id_neighbor):
            if metric == "euclidean":
                dst = np.append(dst, distance.euclidean(X[:,i], X[:,j]))
            elif metric == "squared_correlation":
                dst = np.append(dst, 1 - scipy.stats.pearsonr(X[:,i], X[:,j])[0]**2)
            elif metric == "correlation":
                dst = np.append(dst, 1 - scipy.stats.pearsonr(X[:,i], X[:,j])[0])
            else:
                raise ValueError("Metric '{0}' currently not "
                                     "implemented".format(metric))
    return dst

def get_cluster_dist_mean(X_cluster, size, i, y, method):
    """
    Calculates ward or centroid distance between cluster i and y.
    """
    dst = distance.euclidean(X_cluster[:,i]/size[i], 
                             X_cluster[:,y]/size[y])
    if method == "ward":
        return dst**2/(1/size[i]+1/size[y])
    else:
        return dst

def get_cluster_dist_median(X_cluster, i, y):
    """
    Calculates median euclidean distance between cluster i and y.
    """
    centroid_i = X_cluster[i]
    centroid_y = X_cluster[y]
    return distance.euclidean(centroid_i, centroid_y)
    
def get_cluster_dist_linkage(X, X_cluster, i, z, method, metric, iter_max = 10000):
    """
    Calculates either single, complete or average linkage distance between 
    cluster i and cluster z. In order to avoid a memory error, the distance 
    calculation is partitioned, if at least one of the two clusters includes
    more than iter_max = 10000 variables. 
    """
    voxel_i = X_cluster[i]
    voxel_z = X_cluster[z]
    size_i = len(voxel_i)
    size_z = len(voxel_z)
    if (size_i <= iter_max) and (size_z <= iter_max):
        if metric == "euclidean":
            dst = cdist(X[:,voxel_i].T, X[:,voxel_z].T)
        elif metric == "squared_correlation":
            dst = 1 - (1 - cdist(X[:,voxel_i].T, X[:,voxel_z].T, "correlation"))**2
        elif metric == "correlation":
            dst = cdist(X[:,voxel_i].T, X[:,voxel_z].T, "correlation")
    else:
        n_groups_i = np.ceil(size_i/iter_max).astype(int)
        n_groups_z = np.ceil(size_z/iter_max).astype(int)
        dst = np.array([])
        for j in np.nditer(np.arange(n_groups_i)):
            tmp_vi = voxel_i[(j * iter_max):(np.min(((j + 1) * iter_max, size_i)))]
            for l in np.nditer(np.arange(n_groups_z)):
                tmp_vz = voxel_z[(l * iter_max):(np.min(((l + 1) * iter_max, size_z)))]
                if metric == "euclidean":
                    dst_mat = cdist(X[:,tmp_vi].T, X[:,tmp_vz].T)
                elif metric == "squared_correlation":
                    dst_mat = 1 - (1 - cdist(X[:,tmp_vi].T, X[:,tmp_vz].T, "correlation"))**2 
                elif metric == "correlation":
                    dst_mat = cdist(X[:,tmp_vi].T, X[:,tmp_vz].T, "correlation") 
                if method == "single":
                    dst = np.append(dst, np.min(dst_mat))
                elif method == "complete":
                    dst = np.append(dst, np.max(dst_mat))
                else:
                    dst = np.append(dst, np.sum(dst_mat))     
    if method == "single":
        return np.min(dst)
    elif method == "complete":
        return np.max(dst)
    else:
        return 1/size_i*1/size_z*np.sum(dst)
    
def get_label(Z, V):
    """
    Transforms Z_arr into the desired form. Also works if data set is 
    not contiguous.
    """
    xlim = Z.shape[0]
    dict_labels = dict(zip(range(V), range(V)))
    for i in range(xlim):
        Z1 = int(Z[i,0])
        Z2 = int(Z[i,1])
        Z[i,0] = min(dict_labels[Z1], dict_labels[Z2])
        Z[i,1] = max(dict_labels[Z1], dict_labels[Z2])
        dict_labels[Z2] = int(V + i) 
    return Z 

def get_cluster(Z, V, n_init_cluster):
    """
    Returns vector that indicates the cluster membership of each variable. In 
    total n_init_cluster cluster are generated. Note, that the function even 
    works, if the input data, based on which the linkage matrix is calculated,
    is structured into multiple spatially separated regions of variables.
    
    Parameters
    ----------
    Z : ndarray, shape (V - a, 4)
        Linkage matrix, where 'a' is the number of contiguous regions
        of variables included in the input data.
    V : int
        Number of variables of the input data set, based on which the 
        linkage matrix is genereated.
    n_init_cluster : int
        Number of clusters.

    Returns
    -------
    init_cluster : ndarray, shape (V,)
        Array of integers indicating the cluster membership of each variable, 
        where each variable is assigned to one of in total n_init_cluster 
        clusters.
    """
    
    if n_init_cluster < V - Z.shape[0]:
        n_init_cluster = V - Z.shape[0]
        warnings.warn("n_init_cluster is set to {0} since this is the " 
              "minimum number of contiguous regions in the "
              "data set.".format(V - Z.shape[0]))
    cl = {}
    for i in range(V - n_init_cluster):
        if Z[i, 0] < V and Z[i, 1] < V:
            cl[V + i] = (int(Z[i, 0]), int(Z[i, 1]))
        elif Z[i, 0] < V and not Z[i, 1] < V:
            cl[V + i] = (int(Z[i, 0]), *cl[Z[i, 1]])
            del cl[Z[i, 1]]
        elif not Z[i, 0] < V and Z[i, 1] < V:
            cl[V + i] = (*cl[Z[i, 0]], int(Z[i, 1]))
            del cl[Z[i, 0]]
        else:
            cl[V + i] = (*cl[Z[i, 0]], *cl[Z[i, 1]])
            del cl[Z[i, 0]]
            del cl[Z[i, 1]]
    counter = 1
    init_cluster = np.zeros(V, dtype = int)
    for key in cl:
        init_cluster[list(cl[key])] = counter 
        counter += 1
    del cl
    if counter <= n_init_cluster:
        init_cluster[init_cluster == 0] = range(counter, n_init_cluster + 1)
    return init_cluster


###############################################################################
if __name__ == "__main__" and __name__ != "__main__":
    # Load packages
    # import sys
    # sys.path.append("C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019/Funktionen")
    # import Simulate_spatial_image_data as sid
    from matplotlib import pyplot as plt
    import sklearn.metrics as metrics
    
    ###########################################################################
    #### Test
    """
    Compare with function hierarchy.linkage in python and check if spatial 
    constraint works.
    Check for average, complete, single and centroid linkage
    """
        
    X = np.random.normal(size=48).reshape((6,8))
    matXYZ = np.array([[1,1,1],
                       [1,2,1],
                       [1,3,1],
                       [1,4,1],
                       [2,1,1],
                       [2,2,1],
                       [2,3,1],
                       [2,4,1]])    
    
    # Is spatial constraint fullfilled???
    Z = shac(X, matXYZ, method='average', metric='euclidean',
                          diag_neighbor = False, standardize = False)
    
    plt.figure(figsize=(10, 5))
    plt.title('Spatial Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    # Check
    
    # Under the assumption that all voxels are neighbors to each other, do we 
    # get the same results as in hierarchy.linkage???
    matXYZ_allNeighbors = np.zeros((8,3)) + 1
    Z_allNeighbors = shac(X, matXYZ_allNeighbors, method='centroid', metric='euclidean',
                          diag_neighbor = False, standardize = False)
    
    Z_python = hierarchy.linkage(X.T, method = "centroid", metric = "euclidean")
    
    Z_allNeighbors - Z_python
    # Check 
    
    ###########################################################################
    #### Test 2
    """
    Compare with function hierarchy.linkage in python and check for spatial
    constraint.
    """

    # Generate data, i.e. X and matXYZ
    # Dimensions of images
    vec_dim = (10,10,1)
    # Number of different regions to simulate
    n_regions = 4
    # Number of sample images
    N = 100
    # Generate regions and matXYZ
    vec_cluster, matXYZ = sid.get_regions(vec_dim, n_regions, diag_neighbor = False, rand = None).values()
    # Number of voxels per image
    V = vec_cluster.shape[0]
    # Determine parameters that determine voxel intensities
    vec_mean = np.repeat(0.5, n_regions)
    vec_cor = np.repeat(0.7, n_regions)
    basis_cor = 0.4
    # Determine intensity matrix X
    X, Sigma = sid.get_intensities_normal_correlated(vec_cluster, N = N, vec_mean = vec_mean, 
                                      vec_cor = vec_cor, basis_cor = basis_cor, 
                                      rand = None)
    
    # Under the assumption that all voxels are neighbors to each other, do we 
    # get the same results as in hierarchy.linkage???
    matXYZ_allNeighbors = np.zeros((np.prod(vec_dim), 3)) + 1
    Z_allNeighbors = shac(X, matXYZ_allNeighbors, method='single', metric='euclidean',
                          diag_neighbor = False, standardize = False)
    
    Z_python = hierarchy.linkage(X.T, method = "single", metric = "euclidean")
    
    np.sum(np.abs(Z_allNeighbors[:,(0,1)] - Z_python[:,(0,1)]))
    # Check 
    
    # Does it work well with spatial constraint???
    Z = shac(X, matXYZ, method='average', metric='squared_correlation',
                          diag_neighbor = False, standardize = False)
    pred_cluster = get_cluster(Z, V, n_init_cluster = 4)
    metrics.adjusted_rand_score(vec_cluster, pred_cluster)


