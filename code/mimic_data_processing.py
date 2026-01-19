import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICDataProcessor:
    def __init__(self, data_path=None):
        """
        Initialize the MIMIC data processor
        
        Args:
            data_path (str): Path to the MIMIC data files
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, file_path):
        """
        Load data from a CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, data, feature_columns, target_column):
        """
        Preprocess the data by handling missing values, encoding categorical variables,
        and scaling numerical features
        
        Args:
            data (pd.DataFrame): Input data
            feature_columns (list): List of feature column names
            target_column (str): Name of the target column
            
        Returns:
            tuple: Processed features and target
        """
        try:
            logger.info("Starting data preprocessing")
            self.feature_columns = feature_columns
            self.target_column = target_column
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Encode categorical variables
            data = self._encode_categorical_variables(data)
            
            # Scale numerical features
            data = self._scale_numerical_features(data)
            
            # Split features and target
            X = data[feature_columns]
            y = data[target_column]
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _handle_missing_values(self, data):
        """
        Handle missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        return data
        
    def _encode_categorical_variables(self, data):
        """
        Encode categorical variables using LabelEncoder
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                data[col] = self.label_encoder.fit_transform(data[col])
                
        return data
        
    def _scale_numerical_features(self, data):
        """
        Scale numerical features using StandardScaler
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with scaled numerical features
        """
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if self.target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(self.target_column)
            
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        return data
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: Training and testing sets
        """
        try:
            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise 