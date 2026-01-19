import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICModelTrainer:
    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initialize the model trainer
        
        Args:
            model_type (str): Type of model to train ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm')
            model_params (dict): Dictionary of model parameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the model based on the specified type
        
        Returns:
            object: Initialized model
        """
        models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return models[self.model_type](**self.model_params)
        
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            object: Trained model
        """
        try:
            logger.info(f"Training {self.model_type} model")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            logger.info("Evaluating model performance")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save_model(self, model_path):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
        """
        try:
            logger.info(f"Saving model to {model_path}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            object: Loaded model
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 