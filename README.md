# SPARTACUS
Includes functions to perform spatial hierarchical agglomerative clustering (SHAC),
including SPARTACUS (SPAtial hieRarchical agglomeraTive vAriable ClUStering), 
as well as spatially constrained ensemble clustering (SEC). These functions are 
especially designed to cluster neuroimaging data. Moreover, implementations of 
the silhouette coefficient (SC) for large data, the simplified silhouette 
coefficient (SSC) and spatial adaptations thereof are included as well. 

## Installation

You can install the SPARTACUS package from [PyPI](https://pypi.org/project/SPARTACUS/):

    pip install SPARTACUS

SPARTACUS is supported on Python 3.6 and above.

## How to use

You can call the SHAC and SEC functions in your own Python code, by importing 
from the `SPARTACUS` package:

    >>> from SPARTACUS import spartacus

Example to perform SPARTACUS method, i.e. random input data with V = 24 spatial 
variables on a 4x3x2 grid and ten subjects:
    
    >>> import numpy as np
    >>> V = 24
    >>> X = np.random.normal(size = (10, V))
    >>> matXYZ = np.argwhere(np.zeros((4,3,2)) == 0)
    
SPARTACUS based partition with four clusters:
        
    >>> Z = spartacus.shac(X, matXYZ, metric = 'spartacus', standardize = False)
    >>> labels = spartacus.get_cluster(Z, V, n_init_cluster = 4)
    >>> labels
    array([1, 4, 1, 4, 4, 4, 4, 4, 3, 4, 3, 3, 4, 2, 4, 3, 3, 3, 4, 2, 4, 3, 
           4, 3])
           
Example to perform average linkage based SEC method, i.e. random cluster ensemble 
with V = 8 spatial variables on a 2x2x2 grid and six base partitions:           
    
    >>> import numpy as np    
    >>> X = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,5,5,6,6],
                      [1,1,1,2,3,3,3,4],
                      [1,1,1,2,3,3,3,4]])
    >>> matXYZ = np.argwhere(np.zeros((2,2,2)) == 0)
    
Average linkage based partition with two clusters:
        
    >>> Z = spartacus.spatial_ensemble_clustering(X, matXYZ, method='average')
    >>> labels = spartacus.get_cluster(Z, V = 8, n_init_cluster = 2)
    >>> labels
    array([1, 1, 1, 1, 2, 2, 2, 2])
    

You can call the SC, SSC and spatial adaptations thereof in your own Python code, 
by importing from the `SPARTACUS` package:

    >>> from SPARTACUS import spatial_silhouette

Example evaluation using the silhouette coefficient of randomly generated input 
data with 100 variables and a random partition assigning each variable to one 
of in total four clusters:
    
    >>> import numpy as np
    >>> X = np.random.normal(size = (50, 100))
    >>> labels = np.random.randint(1, 5, 100)
    >>> spatial_silhouette.silhouette_coefficient(X, labels, metric = "euclidean")   
    -0.0171145
    
## Development

To run all the tests run:

    TODO   
    
    
    