"""
Wrapper class for clustering models.
Provides a unified interface for different clustering algorithms.
"""

from abc import ABC, abstractmethod
from joblib import dump, load
from pathlib import Path
import numpy as np
from typing import Optional, Union


class ClusteringWrapper(ABC):
    """
    Abstract base class for clustering model wrappers.
    Provides common interface for all clustering algorithms.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize clustering wrapper.
        
        Parameters:
        -----------
        model_name : str
            Name identifier for the model
        **kwargs : dict
            Model-specific parameters
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.labels_ = None
        self.n_clusters_ = None
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """
        Create the underlying clustering model.
        Must be implemented by subclasses.
        
        Parameters:
        -----------
        **kwargs : dict
            Model-specific parameters
        
        Returns:
        --------
        model : object
            The clustering model instance
        """
        pass
    
    def fit(self, X: np.ndarray) -> 'ClusteringWrapper':
        """
        Fit the clustering model to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        
        Returns:
        --------
        self : ClusteringWrapper
            Returns self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call _create_model first.")
        
        self.model.fit(X)
        self.is_fitted = True
        
        # Extract labels if available
        if hasattr(self.model, 'labels_'):
            self.labels_ = self.model.labels_
        elif hasattr(self.model, 'predict'):
            self.labels_ = self.model.predict(X)
        
        # Extract number of clusters if available
        if hasattr(self.model, 'n_clusters_'):
            self.n_clusters_ = self.model.n_clusters_
        elif hasattr(self.model, 'n_clusters'):
            self.n_clusters_ = self.model.n_clusters
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        
        Returns:
        --------
        labels : np.ndarray
            Cluster labels for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support predict method.")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        
        Returns:
        --------
        labels : np.ndarray
            Cluster labels for each sample
        """
        self.fit(X)
        return self.labels_
    
    def get_labels(self) -> Optional[np.ndarray]:
        """
        Get cluster labels from fitted model.
        
        Returns:
        --------
        labels : np.ndarray or None
            Cluster labels if model is fitted, None otherwise
        """
        return self.labels_
    
    def get_n_clusters(self) -> Optional[int]:
        """
        Get number of clusters.
        
        Returns:
        --------
        n_clusters : int or None
            Number of clusters if available, None otherwise
        """
        return self.n_clusters_
    
    def save(self, filepath: Union[str, Path]) -> 'ClusteringWrapper':
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path where to save the model
        
        Returns:
        --------
        self : ClusteringWrapper
            Returns self for method chaining
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        dump({
            'model': self.model,
            'model_name': self.model_name,
            'labels_': self.labels_,
            'n_clusters_': self.n_clusters_,
            'is_fitted': self.is_fitted
        }, filepath)
        
        return self
    
    def load(self, filepath: Union[str, Path]) -> 'ClusteringWrapper':
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
        
        Returns:
        --------
        self : ClusteringWrapper
            Returns self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        data = load(filepath)
        self.model = data['model']
        self.model_name = data['model_name']
        self.labels_ = data.get('labels_')
        self.n_clusters_ = data.get('n_clusters_')
        self.is_fitted = data.get('is_fitted', False)
        
        return self
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
        --------
        params : dict
            Model parameters
        """
        if self.model is None:
            return {}
        return self.model.get_params() if hasattr(self.model, 'get_params') else {}
    
    def set_params(self, **params) -> 'ClusteringWrapper':
        """
        Set model parameters.
        
        Parameters:
        -----------
        **params : dict
            Parameters to set
        
        Returns:
        --------
        self : ClusteringWrapper
            Returns self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        else:
            for key, value in params.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(model_name='{self.model_name}', status='{status}')"


class KMeansWrapper(ClusteringWrapper):
    """
    Wrapper for sklearn KMeans clustering.
    """
    
    def __init__(self, n_clusters: int = 8, **kwargs):
        """
        Initialize KMeans wrapper.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        **kwargs : dict
            Additional KMeans parameters
        """
        from sklearn.cluster import KMeans
        
        super().__init__(model_name='kmeans', n_clusters=n_clusters, **kwargs)
        self.model = self._create_model(n_clusters=n_clusters, **kwargs)
    
    def _create_model(self, **kwargs):
        """Create KMeans model."""
        from sklearn.cluster import KMeans
        return KMeans(**kwargs)


class DBSCANWrapper(ClusteringWrapper):
    """
    Wrapper for sklearn DBSCAN clustering.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        """
        Initialize DBSCAN wrapper.
        
        Parameters:
        -----------
        eps : float
            Maximum distance between samples in the same neighborhood
        min_samples : int
            Minimum number of samples in a neighborhood
        **kwargs : dict
            Additional DBSCAN parameters
        """
        super().__init__(model_name='dbscan', eps=eps, min_samples=min_samples, **kwargs)
        self.model = self._create_model(eps=eps, min_samples=min_samples, **kwargs)
    
    def _create_model(self, **kwargs):
        """Create DBSCAN model."""
        from sklearn.cluster import DBSCAN
        return DBSCAN(**kwargs)
    
    def get_n_clusters(self) -> Optional[int]:
        """Get number of clusters (excluding noise)."""
        if self.labels_ is not None:
            return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        return None


class AgglomerativeClusteringWrapper(ClusteringWrapper):
    """
    Wrapper for sklearn AgglomerativeClustering.
    """
    
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward', **kwargs):
        """
        Initialize AgglomerativeClustering wrapper.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage : str
            Linkage criterion ('ward', 'complete', 'average', 'single')
        **kwargs : dict
            Additional AgglomerativeClustering parameters
        """
        super().__init__(model_name='agglomerative', n_clusters=n_clusters, linkage=linkage, **kwargs)
        self.model = self._create_model(n_clusters=n_clusters, linkage=linkage, **kwargs)
    
    def _create_model(self, **kwargs):
        """Create AgglomerativeClustering model."""
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(**kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """AgglomerativeClustering doesn't support predict, use fit_predict instead."""
        raise NotImplementedError(
            "AgglomerativeClustering doesn't support predict. Use fit_predict instead."
        )


class ClusteringModelFactory:
    """
    Factory class for creating clustering model wrappers.
    """
    
    _models = {
        'kmeans': KMeansWrapper,
        'dbscan': DBSCANWrapper,
        'agglomerative': AgglomerativeClusteringWrapper,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> ClusteringWrapper:
        """
        Create a clustering model wrapper.
        
        Parameters:
        -----------
        model_type : str
            Type of clustering model ('kmeans', 'dbscan', 'agglomerative')
        **kwargs : dict
            Model-specific parameters
        
        Returns:
        --------
        wrapper : ClusteringWrapper
            Instance of the requested clustering wrapper
        
        Raises:
        -------
        ValueError
            If model_type is not supported
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def list_available_models(cls) -> list:
        """
        List available clustering model types.
        
        Returns:
        --------
        models : list
            List of available model type names
        """
        return list(cls._models.keys())
