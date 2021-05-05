# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:43:29 2018

@author: Tobias Tietz
"""

import warnings
import numpy as np
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics

_ENSEMBLE_LINKAGE_METHODS = {'single': 0, 'complete': 1, 'average': 2, 'centroid': 3,
                    'hellinger': 4}
_ENSEMBLE_DISTANCE_METHODS = ('jaccard')

###############################################################################
##### Spatial ensemble clustering methods
###############################################################################

def spatial_ensemble_clustering(X, matXYZ, method='hellinger', metric='jaccard',
                     diag_neighbor = False, print_progress = False):
    """    
    Perform spatial hierarchical ensemble clustering.
    
    Parameters
    ----------
    X : ndarray, shape(K, V)
        Input ensemble matrix/co-association matrix, where K is the number of 
        different base partitions and V is the number of variables, 
        e.g., voxels. All entries must be positive integers.
    matXYZ : ndarray, shape(V, 3)
        Matrix of variable coordinates. Rows include variable locations sorted 
        as in the column of X, i.e. the coordinates stored in the j-th row 
        of matXYZ correspond to the variable stored in the j-th column of X.
    method : str, optional
        The linkage method. One of 'single', 'complete', 'average', 
        'centroid' or 'hellinger'. Default is 'hellinger'.
    metric : str, optional
        The linkage metric. Only 'jaccard' distance is implemented so far. See 
        ?scipy.spatial.distance.cdist for further details on the 'jaccard' 
        distance. Ignored, if method is equal to 'hellinger'. 
        Default is 'jaccard'.
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors for 
        each voxel. If True, a maximum of 26 voxels belong to each voxels 
        neighborhood. Default is False.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of initializing the sparse ensemble matrix and with every 
        ten thousandth iteration of the main algorithm. Default is False.
        
    Returns
    -------
    Z : ndarray, shape (V - 1, 4)
        Computed linkage matrix.
        
    """   
    # Checks for 'method' and 'metric'     
    if method not in _ENSEMBLE_LINKAGE_METHODS:
        raise ValueError("Invalid method: {0}".format(method))
    if metric not in _ENSEMBLE_DISTANCE_METHODS:
        raise ValueError("Invalid metric: {0}".format(metric))
    # Checks for ensemble matrix
    if X.ndim != 2:
        raise ValueError("X must be 2 dimensional.")
    if X.shape[0] > X.shape[1]:
        warnings.warn('For the two dimensional ensemble matrix the number '
                      'of colums (i.e. the number of variables) is smaller than '
                      'the number of rows (i.e. the number of base partitions). '
                      'You may want to consider the transpose of X instead.')
    if not np.all(np.isfinite(X)):
        raise ValueError("The ensemble matrix must contain only "
                         "finite values.")
    # Checks for matXYZ 
    if matXYZ.ndim != 2:
        raise ValueError("Neighbor matrix matXYZ must be 2 dimensional.")
    if matXYZ.shape[1] != 3:
        raise ValueError("Neighbor matrix matXYZ must have three spatial "
                         "coordinates.")
    if not (np.round(matXYZ)==matXYZ).all():
        raise ValueError("All entries in matXYZ must be integers.")
    # Calculate dendrogram
    Z = ensemble_linkage(X, matXYZ, method, metric, diag_neighbor, print_progress)
    # Return dendrogram
    return Z



def ensemble_linkage(X, matXYZ, method, metric, diag_neighbor = False, 
                     print_progress = False):
    """
    Perform spatial hierarchical clustering.
    
    Parameters
    ----------
    X : ndarray, shape(B, V)
        Input ensemble matrix, where B is the number of different partitions 
        and V is the number of voxels. 
    matXYZ : ndarray, shape(V, 3)
        Matrix of voxel coordinates. Rows include voxel locations sorted 
        as in the column of X.
    method : int
        The linkage method. One of 'single', 'complete', 'average', 
        'centroid' or 'hellinger'.
    metric : str
        The linkage metric. Only 'jaccard' distance is implemented so far. See 
        ?scipy.spatial.distance.cdist for further details on the 'jaccard' 
        distance. Ignored, if method is equal to 'hellinger'.   
    diag_neighbor : bool, optional
        If False, a maximum of six voxels are considered as neighbors for 
        each voxel. If True, a maximum of 26 voxels belong to each voxels 
        neighborhood.
    print_progress : bool, optional
        If True, a progress message is printed with every ten thousandth 
        iteration of initializing the sparse ensemble matrix and with every 
        ten thousandth iteration of the main algorithm. Default is False.
        
    Returns
    -------
    Z : ndarray, shape (V - 1, 4)
        Computed linkage matrix.
        
    """
    # Make entries of ensemble matrix to be 1,2,...,K.
    X = check_ensemble_matrix(X)
    # Number of voxels
    V = matXYZ.shape[0]
    # Initialize clusters, where in the beginning each voxel is its own cluster
    X_cluster = []
    for i in range(V):
        X_cluster.append([i])
    # Initialization of sparse ensemble matrix calculating only distances 
    # between neighbor-variables
    vec_dist = np.array([])
    vec_index1 = np.array([], dtype=np.int)
    vec_index2 = np.array([], dtype=np.int)
    for i in range(V):
        if print_progress:
            if (i % 10000) == 0:
                print("Initial ensemble linkage: Iteration " + str(i))
        if diag_neighbor:
            id_neighbor = np.where(np.max(abs(matXYZ[i,] - matXYZ[(i+1):,]),1) <= 1)[0]+i+1
        else:
            id_neighbor = np.where(np.sum(abs(matXYZ[i,] - matXYZ[(i+1):,]), axis = 1) <= 1)[0]+i+1
        vec_index1 = np.append(vec_index1, np.array([i]*len(id_neighbor), dtype=np.int))
        vec_index2 = np.append(vec_index2, id_neighbor)
        vec_dist = np.append(vec_dist, get_dist_ensemble(X, i, id_neighbor, metric))
    # Main ensemble clustering
    Z_arr = np.empty((V - 1, 4))
    size = np.ones(V, dtype=np.int)  # Sizes of clusters.
    for k in range(V - 1):
        if print_progress:
            if (k % 10000) == 0:
                print("Main ensemble linkage: Iteration " + str(k))
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
        #  Break, if all variables are in the same cluster
        if k == V - 2:
            break
        # Find neighbors of x and y:
        xy_neighbor = np.concatenate((vec_index2[np.where(vec_index1 == x)[0]], 
                                      vec_index2[np.where(vec_index1 == y)[0]],
                                      vec_index1[np.where(vec_index2 == x)[0]],
                                      vec_index1[np.where(vec_index2 == y)[0]]))    
        # Remove x and y from that neighbor list:
        xy_neighbor = xy_neighbor[xy_neighbor != x] 
        xy_neighbor = xy_neighbor[xy_neighbor != y]       
        # Determine unique neighbors and consider if they are neighbors to both 
        # x and y or just to one of them.
        xy_neighbor = np.unique(xy_neighbor)        
        # For centroid or hellinger method update X_cluster already
        if method in ["centroid", "hellinger"]:
            X_cluster[y].extend(X_cluster[x])
        # Determine distance of new cluster to its neighbors
        tmp_vec_index1 = np.array([], dtype = int)
        tmp_vec_index2 = np.array([], dtype = int)
        tmp_vec_dist = np.array([], dtype = int)
        if len(xy_neighbor) > 0:
            for i in np.nditer(xy_neighbor):
                if method in ["centroid", "hellinger"]:
                    K = np.max(X)
                    dst = get_new_dist_ensemble_centroid(X[:, X_cluster[i]], X[:, X_cluster[y]], method, K)
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
                        dist_ix = get_new_dist_ensemble(X, X_cluster, i, x, method, metric)
                    if len(iy) == 1:
                        dist_iy = vec_dist[iy]
                    else:
                        dist_iy = get_new_dist_ensemble(X, X_cluster, i, y, method, metric)
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
        if method not in ["centroid", "hellinger"]:
            X_cluster[y].extend(X_cluster[x])
    # Return linkage matrix    
    Z = get_label(Z_arr, V)
    return Z

def check_ensemble_matrix(X):
    """
    Makes entries of each base partition in ensemble matrix to be 1,2,...,K_b, 
    where b = 1,...,B.
    """
    
    for i in range(X.shape[0]):
        vec_uni_entries = np.unique(X[i,:])
        max_uni_entry = np.max(vec_uni_entries)
        if np.array_equal(vec_uni_entries, np.arange(1,max_uni_entry+1)):
            continue
        elif np.array_equal(vec_uni_entries, np.arange(max_uni_entry)):
            X[i,:] = X[i,:] + 1
        else:
            tmp_row_X = X[i,:].copy()
            counter = 1
            for uni_entry in vec_uni_entries:
                X[i, tmp_row_X==uni_entry] = counter
                counter += 1
    return X


def get_dist_ensemble(X, i, id_neighbor, metric):
    """
    Calculates ensemble distance between voxel i and all voxels in id_neighbor
    """
    dst = np.array([])
    if len(id_neighbor) > 0:
        for j in np.nditer(id_neighbor):
            if metric == "jaccard":
                dst = np.append(dst, 1-metrics.accuracy_score(X[:,i], X[:,j]))
            else:
                raise ValueError("Metric '{0}' currently not "
                                     "implemented".format(metric))
    return dst

def get_new_dist_ensemble_centroid(X_i, X_y, method, K):
    """
    Calculates centroids based or Hellinger based distance between 
    cluster i and cluster y.
    """
    if method == "centroid":
        axis = 1
        # Centroid of cluster i
        u, indices = np.unique(X_i, return_inverse=True)
        centroid_i = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(X_i.shape),
                                        None, np.max(indices) + 1), axis=axis)]
        # Centroid of cluster y
        u, indices = np.unique(X_y, return_inverse=True)
        centroid_y = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(X_y.shape),
                                        None, np.max(indices) + 1), axis=axis)]
        # Return Jaccard distance between centroids
        return 1-metrics.accuracy_score(centroid_i, centroid_y)
    else:
        # Determine probability matrices with dimensions HxK
        D_i = np.apply_along_axis(get_discrete_distribution, 1, X_i, K = K)
        D_y = np.apply_along_axis(get_discrete_distribution, 1, X_y, K = K)
        # Return mean Hellinger distance
        return np.mean(np.sqrt(np.sum((np.sqrt(D_i)-np.sqrt(D_y))**2, 1))/np.sqrt(2))
        
    
def get_discrete_distribution(X_row, K):
    """
    Determine the sample distribution of the base partition stored in X_row, 
    where X_row is a vector with positive integer values that are smaller than 
    or equal to K. 
    """
    # Initiate distribution vector, which includes sample probabilities
    vec_p = np.zeros(K)
    unique, counts = np.unique(X_row, return_counts=True)
    vec_p[unique-1] = counts/X_row.shape[0]
    return vec_p
    

def get_new_dist_ensemble(X, X_cluster, i, z, method, metric, iter_max = 4):
    """
    Calculates linkage based ensemble distance between cluster i and cluster z. 
    """
    voxel_i = X_cluster[i]
    voxel_z = X_cluster[z]
    
    size_i = len(voxel_i)
    size_z = len(voxel_z)
    
    if (size_i <= iter_max) and (size_z <= iter_max):
        if metric == "jaccard":
            dst = cdist(X[:,voxel_i].T, X[:,voxel_z].T, "jaccard")
        else:
            raise ValueError("Metric '{0}' currently not "
                                 "implemented".format(metric))
    else:
        n_groups_i = np.ceil(size_i/iter_max).astype(int)
        n_groups_z = np.ceil(size_z/iter_max).astype(int)
        dst = np.array([])
        for j in np.nditer(np.arange(n_groups_i)):
            tmp_vi = voxel_i[(j * iter_max):(np.min(((j + 1) * iter_max, size_i)))]
            for l in np.nditer(np.arange(n_groups_z)):
                tmp_vz = voxel_z[(l * iter_max):(np.min(((l + 1) * iter_max, size_z)))]
                if metric == "jaccard":
                    dst_mat = cdist(X[:,tmp_vi].T, X[:,tmp_vz].T, "jaccard")
                else:
                    raise ValueError("Metric '{0}' currently not "
                                         "implemented".format(metric))
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
    Transforms Z_arr in desired form. Also works if data set is not contiguous.
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




###############################################################################
if __name__ == "__main__" and __name__ != "__main__":
    
    ###########################################################################
    #### Test 1
    
    from matplotlib import pyplot as plt
    import scipy.cluster.hierarchy as hierarchy
        
    E_mat = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])       
    matXYZ = np.array([[1,1,1],
                       [1,2,1],
                       [1,3,1],
                       [1,4,1],
                       [2,1,1],
                       [2,2,1],
                       [2,3,1],
                       [2,4,1]])    
    
    Z = spatial_ensemble_clustering(E_mat,matXYZ, method = "hellinger")
    
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
    
    ###########################################################################
    #### Test 2
    """
    Test for Ensemble clustering
    """
    import numpy as np
    import os
    import sys
    sys.path.append("C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019/Funktionen")
    import spatial_hierarchical_variable_clusteringV3 as vc
    import Simulate_spatial_image_data as sid
    import matplotlib.pyplot as plt

    #######################################
    # Generate data, i.e. X and matXYZ
    # Dimensions of images
    vec_dim = (12,10,1)
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
    vec_cor = np.repeat(0.6, n_regions)
    basis_cor = 0.4
    # Determine intensity matrix X
    X, Sigma = sid.get_intensities_normal_correlated(vec_cluster, N = N, vec_mean = vec_mean, 
                                      vec_cor = vec_cor, basis_cor = basis_cor, 
                                      rand = None)
    
    ########################################
    #### Generate K clusterings based on Subsamples and calculate ensemble
    #### matrix E
    K = 50
    rand = 31122018
    metric = "euclidean"
    method = "average"
    
    # Initialize ensemble matrix
    E = np.empty((K,V), dtype = int)
    # Initialize array with Adjusted Rand scores
    adj_rand_array = np.empty(K)
    # Generate K clusterings
    for args in range(K):
        # Generate subsample. Therefore:
        # Make results reproduceable by setting a seed
        rn = rand + args
        #np.random.seed(rn)
        id_ss = np.random.choice(np.arange(N, dtype = int), size = int(np.ceil(0.632 * N)), 
                                 replace = False)
        # Generate subsampled data matrix
        X_sub = X[id_ss,:]
        # Make calculations
        Z = vc.hclustvar_spatial(X_sub, matXYZ, method = method, metric = metric,
                                 diag_neighbor = False, standardize = True)
        # Predict clustering
        pred_cluster = vc.get_cluster(Z, V, n_init_cluster = n_regions)
        # Save predicted clustering to ensemble matrix
        E[args,:] = pred_cluster
        # Save Adjusted Rand score between predicted clustering and true clustering
        adj_rand_array[args] = metrics.adjusted_rand_score(vec_cluster, pred_cluster)
        
    ######################################
    #### Compare ensemble partition based on E with vec_cluster
    Z = spatial_ensemble_clustering(E, matXYZ, method = "average")   
    vec_ensemble = vc.get_cluster(Z, V=V, n_init_cluster = n_regions)
    metrics.adjusted_rand_score(vec_ensemble, vec_cluster)
    
    
    ###########################################################################
    # Test 3
    """
    Mean comparison of ensemble clustering with subsample clusterings
    """
    
    #### Calculate spatial ensemble clustering based on E and matXYZ
    # Initiate vector to save Adj Rand scores to
    vec_n_regions = np.arange(2,10)
    vec_adj_rand_ensemble = np.empty(vec_n_regions.shape[0]) 
    vec_mean_adj_rand_ensemble = np.empty(vec_n_regions.shape[0]) 
    # Calculate ensemble dendrogram
    Z = spatial_ensemble_clustering(E, matXYZ, method = "hellinger")
    # Initiate counter
    counter = 0  
    for region in vec_n_regions:
        # Predicted ensemble clustering
        vec_ensemble = vc.get_cluster(Z, V=V, n_init_cluster = region)
        # Adjusted Rand score between ensemble clustering and true clustering
        vec_adj_rand_ensemble[counter] = metrics.adjusted_rand_score(vec_cluster, vec_ensemble)
        tmp_vec = np.empty(E.shape[0])
        for k in range(E.shape[0]):
            tmp_vec[k] = metrics.adjusted_rand_score(vec_ensemble, E[k,:])
        vec_mean_adj_rand_ensemble[counter] = np.mean(tmp_vec)
        counter += 1
    
    # Determine direction to save results to
    os.chdir("C:/Users/admin/Documents/Promotion_Teil_2_MRI/MRI_2019/Plots_paper")
    
    # Plot results
    plt.figure(figsize=(10, 5), dpi=80)
    plt.plot(vec_n_regions, vec_adj_rand_ensemble, "black", 
             label = "Comparison with true clustering")
    plt.plot(vec_n_regions, vec_mean_adj_rand_ensemble, "blue", 
             label = "Mean comparison with subsample clusterings")
    plt.legend()
    plt.xlabel("cluster size")
    plt.ylabel("Adjusted rand index ")
    plt.savefig("Spatial_hierarchical_ensemble_clustering_simulation_" + str(round(vec_cor[0]-basis_cor,2))  
                + "_dim_" + str(vec_dim[0]) + "," + str(vec_dim[1]) + "," + str(vec_dim[2]) + ".pdf")
        
    
    
    
    
    
    
    
