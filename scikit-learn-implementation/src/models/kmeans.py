from sklearn.cluster import KMeans
from .clustering_wrapper import ClusteringWrapper


class KMeansClustering(ClusteringWrapper):
    """
    KMeans clustering model that inherits from ClusteringWrapper.
    Provides a standardized interface for KMeans clustering via the wrapper.

    """
    # Default parameters for KMeans
    DEFAULT_PARAMS = {
        'n_clusters': 8,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'tol': 1e-4,
        'random_state': 42,
        'verbose': 0,
        'copy_x': True,
        'algorithm': 'lloyd'
    }
    
    def __init__(self, n_clusters: int = None, **kwargs):
        """
        Initialize KMeans clustering model.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to form. If not provided, uses default (8)
        **kwargs : dict
            Additional KMeans parameters that override defaults:
            - random_state: int, default=42
            - n_init: int, default=10
            - max_iter: int, default=300
            - tol: float, default=1e-4
            - init: str or array, default='k-means++'
            - verbose: int, default=0
            - copy_x: bool, default=True
            - algorithm: str, default='lloyd'
        """
        # Use provided n_clusters or default
        if n_clusters is not None:
            kwargs['n_clusters'] = n_clusters
        
        super().__init__(model_name="kmeans", **kwargs)
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """
        Create the underlying KMeans model with default parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            KMeans parameters (will override defaults)
        
        Returns:
        --------
        model : KMeans
            KMeans clustering model instance
        """
        params = self.DEFAULT_PARAMS.copy()
        
        # Update with any provided kwargs (user parameters override defaults)
        params.update(kwargs)
        
        return KMeans(**params)