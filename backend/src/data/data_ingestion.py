import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    with open("params.yaml","r") as f:
        params = yaml.safe_load(f)
        logger.info("Successfully loaded parameters from params.yaml")
except Exception as e:
    logger.error(f"Error loading params.yaml: {e}")
    raise

def data_ingestion(file_path):
    """"
    Ingests data from a CSV file, splits it into training and testing sets, and returns them.
    
    Args:
        file_path (str): The path to the CSV file containing the data."""
    
    try:
        # load data
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

        df = df.dropna()  # Drop rows with missing values
        logging.info(f"Data after dropping missing values. Shape: {df.shape}")

        df = df.drop_duplicates()  # Drop duplicate rows
        logging.info(f"Data after dropping duplicates. Shape: {df.shape}")

        df = df.drop(columns = ['id' , 'date'], axis = 1)  # Drop unnecessary columns
        logging.info(f"Data after dropping unnecessary columns. Shape: {df.shape}")

        logging.info(f"Data ingestion completed successfully. Final shape: {df.shape}")

        return df
    except Exception as e:
        logging.error(f"Error occurred while ingesting data: {e}")
        raise


def split_data(df , test_size, random_state):
    """"splitts the data into training and testing sets based on parameters defined in params.yaml"""
    try:
        X = df.drop(columns=['price'], axis=1)
        
        y = np.log1p(df['price'])  # Log-transform the target variable to handle skewness
        logging.info(f"Data split into features and target variable. Features shape: {X.shape}, Target shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into training and testing sets. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error occurred while splitting data: {e}")
        raise

def save_data(data , file_path):
    """""Save the training and testing sets to CSV files in the specified output directory."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving data: {e}")
        raise



def main():
    """"Main function to execute the data ingestion and splitting process."""
    try:
        file_path = os.path.join("kc_house_data.csv")
        df = data_ingestion(file_path)
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

        save_data(X_train, os.path.join("data","processed","X_train.csv"))
        save_data(X_test, os.path.join("data","processed","X_test.csv"))
        save_data(y_train, os.path.join("data","processed","y_train.csv"))
        save_data(y_test, os.path.join("data","processed","y_test.csv"))

    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise

if __name__ == "__main__":
    main()