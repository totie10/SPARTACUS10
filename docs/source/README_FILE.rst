Description of the methods
==========================

SHAC and SEC methods 
--------------------

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

The SHAC and SEC methods are implemented in the :code:`spartacus` module from the :code:`SPARTACUS`
package.


Silhouette coefficient and its adaptations
------------------------------------------

In order to evaluate the quality of the final parcellations, internal validation
measures can be employed, one of which is the well established silhouette 
coefficient (SC) (Rousseeuw, 1987). 

Let :math:`\mathbf{C}_K=(C_1,\ldots,C_K)` be a parcellation which should be evaluated 
on a data set :math:`\mathbf{X}\in \mathbb{R}^{N\times V}`, where :math:`N` is the number of 
subjects and :math:`V` is the number of image voxels. The silhouette width of a single 
voxel :math:`\mathbf{x}_j, j=1,\ldots,V,` belonging to cluster :math:`C_k, k=1,\ldots,K`
is given by

    .. math::
    
       s_j = \dfrac{b_j-a_j}{\max\{a_j,b_j\}},

where

    .. math::

       a_j = \dfrac{1}{|C_k|-1}\sum\limits_{\substack{\mathbf{x}_\ell \in C_k \\ \ell\neq j}}d\big(\mathbf{x}_j, \mathbf{x}_\ell\big)
       
is the average distance of :math:`\mathbf{x}_j` to all other voxels in :math:`C_k` and 

    .. math::
    
       b_j = \min_{m \neq k}\dfrac{1}{|C_m|}\sum_{\mathbf{x}_\ell \in C_m}d\big(\mathbf{x}_j, \mathbf{x}_\ell\big)
       
is the average distance of :math:`\mathbf{x}_j` to all voxels in the closest cluster.  
As distance :math:`d`, e.g., the Euclidean distance or the correlation based distance

    .. math::
    
       d_\text{corr}(\mathbf{x}_j,\mathbf{x}_\ell)=1-|\text{corr}(\mathbf{x}_j,\mathbf{x}_\ell)|

can be employed. The silhouette coefficient of :math:`\mathbf{C}_K` is then given as

    .. math::
    
       \text{SC} = \dfrac{1}{N}\sum_{j=1}^Vs_j.

An issue with the silhouette coefficient is that its memory consuming, if the 
number of voxels :math:`V` is large, as it is typically the case with high-resolution 
neuroimages. In order to avoid running into a memory error, the module 
:code:`spatial_silhouette` from the :code:`SPARTACUS` package includes an implementation 
of the silhouette coefficient for large data sets that avoids to run into a memory 
error, by calculating the pairwise distances between voxels not all at once, but 
in chunks. 

Another issue of the silhouette coefficient is that it is computationally expensive,
if :math:`V` is large. Therefore, the computationally less expensive simplified silhouette 
coefficient (SSC) as introduced by Vendramin et al. (2010) is implemented in the
spatial_silhouette module as well.
In this variation, :math:`a_j` is the distance of voxel :math:`\mathbf{x}_j` to the 
centroid :math:`\mathbf{c}_k \in \mathbb{R}^V` of its cluster :math:`C_k`, i.e.

    .. math::

       a_j = d(\mathbf{x}_j, \mathbf{c}_k)

and :math:`b_j` is the minimum of the distances of :math:`\mathbf{x}_j` to the centroids 
of the other clusters, i.e.

    .. math::

       b_j = \min_{m\neq k}d(\mathbf{x}_j,  \mathbf{c}_m).
        
The distance measure :math:`d` is either the Euclidean distance or the correlation
distance :math:`d_\text{corr}` and the centroid is either the mean over all data points 
in :math:`C_k` or the first principal component of that cluster, respectively.   

Both, the silhouette coefficient and its simplified variant ignore the spatial 
information provided by the input data. However, cross-hemispheric communications,
i.e. interactions between contralateral regions on different brain hemispheres
(Davis and Cabeza, 2015), can cause the concerned brain regions to be of high 
similarity. As these brain regions are usually spatially discontiguous, they can
not be merged by a spatial clustering algorithm and, therefore, reduce the
SC or SSC score. Therefore, spatial adaptations of the SC and SSC as proposed 
by Tietz et al. (2021) are implemented in the :code:`spatial_silhouette` module that are 
independent of cross-hemispheric communications. The idea is to calculate 
the :math:`b_j` value of voxel :math:`\mathbf{x}_j` belonging to cluster :math:`C_k` 
only with respect to the neighboring clusters of :math:`C_k`. 

More precisely, clusters 
:math:`C_m` and :math:`C_k` are defined to be neighbors, if and only if :math:`s_{km}^*=1`, where

    .. math::
     
       s_{km}^*= \text{I}\left(\sum_{\mathbf{x}_j\in C_k}\sum_{\mathbf{x}_\ell\in C_m}s_{j\ell}>0\right)

and :math:`s_{j\ell}=1`, if :math:`\mathbf{x}_j` and :math:`\mathbf{x}_\ell` 
are neighbors, otherwise, :math:`s_{j\ell}=0`. 
The modified :math:`b_j`-value 

    .. math::
       b_j^\text{spatial}=\min_{\substack{m\neq k \\s_{km}^*=1}}\dfrac{1}{|C_m|}\sum_{\mathbf{x}_\ell \in C_m}d\big(\mathbf{x}_j, \mathbf{x}_\ell\big)

of :math:`\mathbf{x}_j \in C_k` is used to calculate the spatial SC, referred to as 
:math:`SC_\text{spatial}`, and the modified :math:`b_j`-value 

    .. math:: 
       b_j^\text{spatial} = \min_{\substack{m\neq k \\s_{km}^*=1}}d(\mathbf{x}_j,  \mathbf{c}_m),

of :math:`\mathbf{x}_j \in C_k` can be used to calculate the spatial SSC, 
referred to as :math:`SSC_\text{spatial}`.


Installation
============

You can install the SPARTACUS package from [PyPI](https://pypi.org/project/SPARTACUS/)::

    pip install SPARTACUS

SPARTACUS is supported on Python 3.6 and above.


How to use
==========

You can call the SHAC and SEC functions in your own Python code, by importing 
from the :code:`SPARTACUS` package::

    from SPARTACUS import spartacus

Example to perform SPARTACUS method, i.e. random input data with V = 24 spatial 
variables on a 4x3x2 grid and ten subjects::
    
    >>> import numpy as np
    >>> V = 24
    >>> X = np.random.normal(size = (10, V))
    >>> matXYZ = np.argwhere(np.zeros((4,3,2)) == 0)
    
SPARTACUS based partition with four clusters::
        
    >>> Z = spartacus.shac(X, matXYZ, metric = 'spartacus', standardize = False)
    >>> labels = spartacus.get_cluster(Z, V, n_init_cluster = 4)
    >>> labels
    array([1, 4, 1, 4, 4, 4, 4, 4, 3, 4, 3, 3, 4, 2, 4, 3, 3, 3, 4, 2, 4, 3, 
           4, 3])
           
Example to perform average linkage based SEC method, i.e. random cluster ensemble 
with V = 8 spatial variables on a 2x2x2 grid and six base partitions::           
    
    >>> import numpy as np    
    >>> X = np.array([[1,1,2,2,3,3,4,4],
    >>>               [1,1,2,2,3,3,4,4],
    >>>               [1,1,2,2,3,3,4,4],
    >>>               [1,1,2,2,5,5,6,6],
    >>>               [1,1,1,2,3,3,3,4],
    >>>               [1,1,1,2,3,3,3,4]])
    >>> matXYZ = np.argwhere(np.zeros((2,2,2)) == 0)
    
Average linkage based partition with two clusters::
        
    >>> Z = spartacus.spatial_ensemble_clustering(X, matXYZ, method='average')
    >>> labels = spartacus.get_cluster(Z, V = 8, n_init_cluster = 2)
    >>> labels
    array([1, 1, 1, 1, 2, 2, 2, 2])
    

You can call the SC, SSC and spatial adaptations thereof in your own Python code, 
by importing from the :code:`SPARTACUS` package::

    >>> from SPARTACUS import spatial_silhouette

Example evaluation using the silhouette coefficient of randomly generated input 
data with 100 variables and a random partition assigning each variable to one 
of in total four clusters::
    
    >>> import numpy as np
    >>> X = np.random.normal(size = (50, 100))
    >>> labels = np.random.randint(1, 5, 100)
    >>> spatial_silhouette.silhouette_coefficient(X, labels, metric = "euclidean")   
    -0.0171145
    
  
Development
===========

To run all the tests run::

    py.test 

from the SPARTACUS directory (installed in side_packages). 
    
References
==========

Carvalho AXY, Albuquerque PHM, de Almeida Junior GZ, Guimaraes RD (2009)
Spatial hierarchical clustering. Revista Brasileira de Biometria 27(3):411-442
        
Vigneau E, Qannari EM (2003) Clustering of variables around latent components.
Communications in Statistics-Simulation and Computation 32(4):1131-1150

Rousseeuw PJ (1987) Silhouettes: a graphical aid to the interpretation and 
validation of cluster analysis. Journal of computational and applied mathematics 20:53-65
        
Vendramin L, Campello RJGB, Hruschka ER (2010) Relative clustering validity 
criteria: A comparative overview. Statistical analysis and data mining: 
the ASA data science journal 3(4):209-235
        
Davis SW, Cabeza R (2015) Cross-hemispheric collaboration and segregation associated
with task difficulty as revealed by structural and functional connectivity.
Journal of Neuroscience 35(21):8191-8200
        
Tietz et al. (2021) (Publication in progress.)    
    