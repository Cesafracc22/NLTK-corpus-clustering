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
    
