# SPARTACUS
Includes functions to perform spatial hierarchical agglomerative clustering (SHAC),
including SPARTACUS (SPAtial hieRarchical agglomeraTive vAriable ClUStering), 
as well as spatially constrained ensemble clustering (SEC). These functions are 
especially designed to cluster neuroimaging data. Moreover, implementations of 
the silhouette coefficient (SC) for large data, the simplified silhouette 
coefficient (SSC) and spatial adaptations thereof are included as well. 

## Description of the methods

### SHAC and SEC methods 

In neuroimaging, a goal is to subdivide the human brain into spatially contiguous 
parcels based on, e.g., structural or functional brain images.
These parcels, i.e. brain regions, should be of large within homogeneity and between 
heterogeneity with respect to the underlying modality that these 
parcels are derived from, but, ideally, also with respect to other modalities.

To achieve this goal, clustering methods are typically employed. One group of 
clustering methods paticularly suited for such parcellation tasks are SHAC methods
(Carvalho et al., 2009). 
The main idea is to, in the beginning, consider each image voxel as its own cluster
and to merge in each step of the algorithm the clostest two clusters under the
constraint that these clusters must be spatial neighbors. Hereby, two clusters are 
spatial neighbors, if at least one voxel from one cluster is a spatial neighbor 
of a voxel from the other cluster. The alorithm proceeds until either all voxels are 
in the same cluster or until all clusters are discontiguous. The latter occurs,
if the input data consists of multiple discontiguous regions. In order to calculate
the distance between clusters, popular linkage functions such as single, complete 
or average linkage as well as, e.g. the centroid or Ward's minimal variance method
can be employed.

A SHAC method that is especially designed to cluster variables, i.e. voxels, is the SPARTACUS 
(SPAtial hieRarchical agglomeraTive vAriable ClUStering) method (Tietz et al, 2021).
The SPARTACUS distance between two clusters is the overall loss in explained 
total variance by all clusters first principal components that would be caused, 
if these two clusters are merged (also see Vigneau and Qannari (2003)).

Some SHAC methods, e.g. single, complete or average linkage based SHAC, can also 
be used to perform spatial hierarchical ensemble clustering (SEC) as described 
by Tietz et al. (2021). Hereby, the SHAC method is applied to the co-association/ensemble
matix, which is calculated from a cluster ensemble. Moreover, also the Hellinger 
distance can be used as distance measure for clusters in a SHAC algorithm as 
introduced by Tietz et al. (2021), in order to obtain a final ensemble parcellation 
based on a cluster ensemble. 

The SHAC and SEC methods are implemented in the spartacus module from the SPARTACUS
package.

### Silhouette coefficient and its adaptations

In order to evaluate the quality of the final parcellations, internal validation
measures can be employed, one of which is the well established silhouette 
coefficient (SC) (Rousseeuw, 1987). The silhouette coefficient of a voxel is 
calculated as ``(b - a) / max(a, b)``, where ``a`` is the mean distance of that 
voxel to all other voxels from its cluster and ``b`` is the smallest mean distance 
of that voxel to all voxels from another cluster. As distance, e.g., the Euclidean
distance or the correlation distance (1 - abs(corr)) is used. The silhouette 
coefficient of a parcellation is then the mean over all these voxel-wise 
silhouette coefficients.

An issue with the silhouette coefficient is that its memory consuming, if the 
number of voxels is large, as it is typically the case with high-resolution 
neuroimages. In order to avoid running into a memory error, the module 
spatial_silhouette from the SPARTACUS package includes an implementation 
of the silhouette coefficient for large data sets that avoids to run into a memory 
error, by calculating the pairwise distances between voxels not all at once, but 
in chunks. 

Another issue of the silhouette coefficient is that it is computationally expensive,
if the number of voxels is large. Therefore, the computationally less expensive 
simplified silhouette coefficient (SSC) as introduced by Vendramin et al. (2010) 
is implemented in spatial_silhouette module as well.
In this variation, ``a`` is the distance of the respective voxel to the centroid
of its cluster and ``b`` is the smallest distance of that voxel to the centroid 
of another cluster. The distance measure is either the Euclidean distance or the correlation
distance and the centroid of a cluster is either the mean over all data points 
from that cluster or the first principal component of that cluster, respectively.   

Both, the silhouette coefficient and its simplified variant ignore the spatial 
information provided by the input data. However, cross-hemispheric communications,
i.e. interactions between contralateral regions on different brain hemispheres
(Davis and Cabeza, 2015), can cause the concerned brain regions to be of high 
similarity. As these brain regions are usually spatially discontiguous, they can
not be merged by a spatial clustering algorithm and, therefore, reduce the
SC or SSC score. Therefore, spatial adaptations of the SC and SSC as proposed 
by Tietz et al. (2021) are implemented in the spatial_silhouette module that are 
independent of cross-hemispheric communications. The idea is to calculate 
the ``b`` value of each voxel only with respect to the neighboring clusters of 
the cluster this voxel belongs to. 


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

To run all the tests run

    py.test 

from the SPARTACUS directory (installed in side_packages). 
    
## References

Carvalho AXY, Albuquerque PHM, de Almeida Junior GZ, Guimaraes RD (2009)
        Spatial hierarchical clustering. Revista Brasileira de Biometria 
        27(3):411-442
        
Vigneau E, Qannari EM (2003) Clustering of variables around latent components.
        Communications in Statistics-Simulation and Computation 32(4):1131-1150

Rousseeuw PJ (1987) Silhouettes: a graphical aid to the interpretation and 
        validation of cluster analysis. Journal of computational and applied 
        mathematics 20:53-65
        
Vendramin L, Campello RJGB, Hruschka ER (2010) Relative clustering validity 
        criteria: A comparative overview. Statistical analysis and data mining: 
        the ASA data science journal 3(4):209-235
        
Davis SW, Cabeza R (2015) Cross-hemispheric collaboration and segregation associated
        with task diculty as revealed by structural and functional connectivity.
        Journal of Neuroscience 35(21):8191-8200
        
Tietz et al. (2021) (Publication in progress.)    
    