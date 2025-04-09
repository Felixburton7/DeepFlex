"""
Random Forest model implementation for the DeepFlex ML pipeline.

This module provides a RandomForestModel for protein flexibility prediction
with support for uncertainty estimation and hyperparameter optimization.
"""

# /home/s_felix/DeepFlex/deepflex/models/random_forest.py

# ... (keep imports and __init__) ...
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import inspect

from deepflex.models import register_model
from deepflex.models.base import BaseModel
from deepflex.utils.helpers import ProgressCallback, ensure_dir # Added ensure_dir

logger = logging.getLogger(__name__)

@register_model("random_forest")
class RandomForestModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float, int] = 0.7, 
        bootstrap: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        # Filter kwargs to only include valid RandomForestRegressor parameters
        rf_params = set(inspect.signature(RandomForestRegressor).parameters.keys())
        self.model_params = {k: v for k, v in kwargs.items() if k in rf_params}
        # Store hyperparameter search config separately if present in kwargs
        self.randomized_search_config = kwargs.get('randomized_search', {})

        self.model = None
        self.feature_names_ = None
        self.best_params_ = None # Store best params if HPO is used


        # Place this corrected method inside the RandomForestModel class
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], feature_names: Optional[List[str]] = None) -> 'RandomForestModel':
        """
        Train the Random Forest model. Includes timing and verbose HPO output.

        Args:
            X: Feature matrix
            y: Target RMSF values
            feature_names: Optional list of feature names

        Returns:
            Self, for method chaining
        """
        # Store feature names
        if isinstance(X, pd.DataFrame): self.feature_names_ = X.columns.tolist()
        elif feature_names is not None: self.feature_names_ = feature_names
        else: self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure y is 1D numpy array
        if isinstance(y, pd.Series): y = y.values
        y = np.ravel(y) # Convert to 1D array

        # Check if randomized search is enabled
        use_randomized_search = self.randomized_search_config.get('enabled', False)

        fit_start_time = time.time() # Time the whole fit process

        if use_randomized_search:
            logger.info("RandomizedSearchCV enabled for Random Forest training.")
            search_params = self.randomized_search_config
            n_iter = search_params.get('n_iter', 10) # Default to 10 iterations if not set
            cv = search_params.get('cv', 3)       # Default to 3 folds if not set
            # --- Get verbose level from config, default to 1 for some output ---
            verbose_level = search_params.get('verbose', 1)
            param_distributions = search_params.get('param_distributions')

            if not isinstance(param_distributions, dict) or not param_distributions:
                logger.error("HPO enabled, but 'param_distributions' is missing or invalid in config. Aborting train.")
                raise ValueError("Invalid 'param_distributions' for RandomizedSearchCV in config.")

            # Base estimator uses params from __init__/config, but HPO overrides these
            base_rf = RandomForestRegressor(
                 # Set core params that are NOT searched here if needed,
                 # otherwise let the search find them.
                 random_state=self.random_state,
                 # Pass other non-searched valid RF params stored during init
                 **self.model_params
            )

            logger.info(f"Setting up RandomizedSearchCV with n_iter={n_iter}, cv={cv}, verbose={verbose_level}")
            search = RandomizedSearchCV(
                estimator=base_rf,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error', # Or other appropriate score
                n_jobs=self.model_params.get('n_jobs', -1),
                random_state=self.random_state,
                verbose=verbose_level, # <-- USE VERBOSE LEVEL
                return_train_score=False
            )

            # Fit the randomized search - verbose output goes to stdout/stderr
            logger.info("Fitting RandomizedSearchCV (verbose output below)...")
            search.fit(X, y) # Remove the ProgressCallback wrapper here
            # Verbose output from scikit-learn will be printed directly

            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            logger.info(f"RandomizedSearchCV complete. Best Score ({search.scorer_.__name__}): {search.best_score_:.4f}")
            logger.info(f"Best hyperparameters found: {self.best_params_}")

        else:
            # --- Standard Training (No RandomizedSearch) ---
            logger.info("Standard Random Forest training (no hyperparameter search).")
            # Use ProgressCallback for the single fit operation
            with ProgressCallback(total=1, desc="Training Random Forest") as pbar:
                rf_init_params = {
                     'n_estimators': self.n_estimators, 'max_depth': self.max_depth,
                     'min_samples_split': self.min_samples_split, 'min_samples_leaf': self.min_samples_leaf,
                     'max_features': self.max_features, 'bootstrap': self.bootstrap,
                     'random_state': self.random_state,
                     **self.model_params # Add other valid RF params
                }
                self.model = RandomForestRegressor(**rf_init_params)
                self.model.fit(X, y)
                pbar.update()
            # Log total time for standard fit outside the progress bar scope
            fit_end_time = time.time()
            logger.info(f"Standard RF fitting completed in {fit_end_time - fit_start_time:.2f} seconds.")


        # Log total time for the entire fit method (includes HPO if run)
        total_fit_time = time.time() - fit_start_time
        logger.info(f"RandomForestModel fit method finished in {total_fit_time:.2f} seconds.")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None: raise RuntimeError("Model must be trained before prediction.")
        try:
            # Ensure estimators_ attribute exists and is populated
            if not hasattr(self.model, 'estimators_') or not self.model.estimators_:
                 logger.warning("Model has no estimators_ attribute or it's empty. Cannot calculate uncertainty.")
                 # Fallback: return mean prediction and zero uncertainty
                 mean_prediction = self.model.predict(X)
                 return mean_prediction, np.zeros_like(mean_prediction)

            all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            mean_prediction = np.mean(all_preds, axis=0)
            std_prediction = np.std(all_preds, axis=0)
            return mean_prediction, std_prediction
        except Exception as e:
            logger.error(f"Error during uncertainty prediction: {e}. Falling back.")
            mean_prediction = self.model.predict(X)
            return mean_prediction, np.zeros_like(mean_prediction)


    def save(self, path: str) -> None:
        if self.model is None: raise RuntimeError("Cannot save untrained model.")
        ensure_dir(os.path.dirname(path)) # Use helper
        state = {
            'model': self.model,
            'feature_names': self.feature_names_,
            'best_params': self.best_params_,
            # Store init params separately for potential re-instantiation if needed
            'init_params': {
                'n_estimators': self.n_estimators, 'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split, 'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features, 'bootstrap': self.bootstrap,
                'random_state': self.random_state, **self.model_params
            },
             'randomized_search_config': self.randomized_search_config # Save search config too
        }
        try:
             joblib.dump(state, path)
             logger.info(f"Random Forest model saved to {path}")
        except Exception as e:
             logger.error(f"Failed to save Random Forest model to {path}: {e}", exc_info=True)

    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found: {path}")
        try:
            state = joblib.load(path)
            # Re-instantiate using saved init params if needed, or just load the model object
            # For scikit-learn models, loading the object directly is usually fine
            instance = cls() # Create a blank instance first
            instance.model = state['model']
            instance.feature_names_ = state.get('feature_names')
            instance.best_params_ = state.get('best_params')
            # Restore init params and config for reference
            init_params = state.get('init_params', {})
            instance.n_estimators = init_params.get('n_estimators', 100)
            instance.max_depth = init_params.get('max_depth')
            instance.min_samples_split = init_params.get('min_samples_split', 2)
            instance.min_samples_leaf = init_params.get('min_samples_leaf', 1)
            instance.max_features = init_params.get('max_features', 0.7)
            instance.bootstrap = init_params.get('bootstrap', True)
            instance.random_state = init_params.get('random_state', 42)
            instance.model_params = {k: v for k, v in init_params.items() if k not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']}
            instance.randomized_search_config = state.get('randomized_search_config', {})

            logger.info(f"Random Forest model loaded from {path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading Random Forest model from {path}: {e}", exc_info=True)
            raise

    def get_feature_importance(self, X_val=None, y_val=None, method="permutation", n_repeats=10) -> Optional[Union[Dict[str, float], np.ndarray]]:
         if self.model is None: return None
         if X_val is not None and y_val is not None and method == "permutation":
             try:
                 from sklearn.inspection import permutation_importance
                 logger.debug(f"Calculating permutation importance with {n_repeats} repeats...")
                 r = permutation_importance(
                     self.model, X_val, y_val, n_repeats=n_repeats,
                     random_state=self.random_state, n_jobs=self.model_params.get('n_jobs', -1)
                 )
                 logger.debug("Permutation importance calculation finished.")
                 return r.importances_mean # Return array, mapping done in Pipeline.analyze
             except Exception as e:
                 logger.warning(f"Permutation importance failed: {e}. Falling back to impurity-based.")
                 method = "impurity" # Fallback

         if method == "impurity":
             try:
                 logger.debug("Using impurity-based feature importance.")
                 importances = self.model.feature_importances_
                 if self.feature_names_ and len(self.feature_names_) == len(importances):
                     return dict(zip(self.feature_names_, importances))
                 else: # Return as array if names mismatch or missing
                     return importances
             except Exception as e:
                 logger.warning(f"Could not get impurity-based importance: {e}")
                 return None
         else:
              logger.warning(f"Unsupported feature importance method: {method}")
              return None

    def hyperparameter_optimize(self, X, y, param_grid, method="random", n_trials=20, cv=3):
        # This method should really just call fit with the right search params enabled
        logger.info("Hyperparameter optimization requested for RF. Running fit with RandomizedSearchCV.")
        if not self.randomized_search_config:
             logger.warning("HPO requested, but 'randomized_search' config missing. Setting up defaults.")
             self.randomized_search_config = {'enabled': True, 'n_iter': n_trials, 'cv': cv, 'param_distributions': param_grid}
        else:
             self.randomized_search_config['enabled'] = True
             self.randomized_search_config['n_iter'] = n_trials
             self.randomized_search_config['cv'] = cv
             self.randomized_search_config['param_distributions'] = param_grid

        self.fit(X, y) # Re-run fit, which will now use RandomizedSearchCV
        return self.best_params_ if self.best_params_ else {}
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate RMSF predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted RMSF values
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate RMSF predictions with standard deviation (uncertainty).
        
        Uses the variance of predictions across the ensemble of trees
        as a measure of prediction uncertainty.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, std_dev) arrays
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Make predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def hyperparameter_optimize(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        method: str = "bayesian",
        n_trials: int = 20,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid or distributions
            method: Optimization method ("grid", "random", or "bayesian")
            n_trials: Number of trials for random or bayesian methods
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        # Random Forest ignores the method and n_trials parameters, using RandomizedSearchCV instead
        if method != "random":
            logger.warning(f"RandomForest only supports 'random' method for optimization, ignoring '{method}'")
            
        with ProgressCallback(total=1, desc="Hyperparameter optimization") as pbar:
            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                n_iter=n_trials,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
                return_train_score=True
            )
            
            search.fit(X, y)
            pbar.update()
            
        # Update model with the best estimator
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
        logger.info(f"Best hyperparameters: {self.best_params_}")
        
        return self.best_params_
        
    def save(self, path: str) -> None:
        """
        Save model to disk using joblib.
        
        Args:
            path: Path to save location
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state
        state = {
            'model': self.model,
            'feature_names': self.feature_names_,
            'best_params': self.best_params_,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'random_state': self.random_state,
                'model_params': self.model_params
            }
        }
        
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded RandomForestModel instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            state = joblib.load(path)
            
            # Create new instance with saved parameters
            params = state['params']
            instance = cls(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 0.7),
                bootstrap=params.get('bootstrap', True),
                random_state=params.get('random_state', 42),
                **params.get('model_params', {})
            )
            
            # Restore model and feature names
            instance.model = state['model']
            instance.feature_names_ = state.get('feature_names', None)
            instance.best_params_ = state.get('best_params', None)
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
    def get_feature_importance(self, X_val=None, y_val=None) -> Dict[str, float]:
        """
        Get feature importance values using permutation importance.
        
        Args:
            X_val: Optional validation features for permutation importance
            y_val: Optional validation targets for permutation importance
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.model is None:
            return {}
        
        # If validation data is provided, use permutation importance
        if X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                from sklearn.inspection import permutation_importance
                
                # Calculate permutation importance
                r = permutation_importance(
                    self.model, X_val, y_val, 
                    n_repeats=10, 
                    random_state=self.random_state
                )
                
                # Use mean importance as the feature importance
                importance_values = r.importances_mean
                
                # Map to feature names if available
                if self.feature_names_ is not None and len(self.feature_names_) == len(importance_values):
                    return dict(zip(self.feature_names_, importance_values))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importance_values)}
                    
            except Exception as e:
                logger.warning(f"Could not compute permutation importance: {e}")
                # Fall back to built-in feature importance
        
        # Use built-in feature importance as fallback
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            
            if self.feature_names_ is not None and len(self.feature_names_) == len(importance_values):
                return dict(zip(self.feature_names_, importance_values))
            else:
                return {f"feature_{i}": importance for i, importance in enumerate(importance_values)}
        
        return {}