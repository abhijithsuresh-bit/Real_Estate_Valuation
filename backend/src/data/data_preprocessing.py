import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
import yaml
from sklearn.cluster import KMeans

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger  = logging.getLogger(__name__)

try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        logging.info("Successfully loaded parameters from params.yaml")
except Exception as e:
    logging.error(f"Error occurred while loading parameters: {e}")


def data_preprocessing(file_path):

    try:
        """" Preprocessesing the data by feature engineering and scaling the features based on parameters defined in params.yaml"""
        logging.info("Starting data preprocessing...")

        df = pd.read_csv(file_path)

        df["house_age"] = 2026 - df["yr_built"]  # Feature engineering: Calculate house age
        logging.info("Feature engineering completed: 'house_age' column created.")

        df["is_renovated"]  = (df["yr_renovated"]>0).astype(int)  # Feature engineering: Create a binary feature for renovation status
        logging.info("Feature engineering completed: 'is_renovated' column created.")

        df['year_since_renovation'] = np.where(
            df['yr_renovated'] == 0, 0 , 2026 - df["yr_renovated"])  # Feature engineering: Calculate years since renovation
        logging.info("Feature engineering completed: 'year_since_renovation' column created.")
    
        df["total_sqft"] = df["sqft_living"] + df["sqft_basement"]  # Feature engineering: Create a new feature for total square footage
        logging.info("Feature engineering completed: 'total_sqft' column created.")
    
        # "How large is the house relative to the land?" simply means if we have a plot of 10 cents 5 cent is house that we are calculating
        df["living_to_lot_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)  # Feature engineering: Create a new feature for living area to lot size ratio
        logging.info("Feature engineering completed: 'living_to_lot_ratio' column created.")

        df["size_vs_neighbor"] = df["sqft_living"] - df["sqft_living15"]  # Feature engineering: Create a new feature comparing the size of the house to its neighbors
        logging.info("Feature engineering completed: 'size_vs_neighbor' column created.")

        df["location_cluster"] = KMeans(n_clusters=50, random_state=42).fit_predict(df[["lat", "long"]])  # Feature engineering: Create a new feature for location clusters using KMeans
        logging.info("Feature engineering completed: 'location_cluster' column created using KMeans clustering.")

        df["quality_score"] = df["grade"] * df["condition"]  # Feature engineering: Create a new feature for overall quality score
        logging.info("Feature engineering completed: 'quality_score' column created.")

        df["is_luxury"] = ((df["grade"] >= 10) & (df["waterfront"] == 1)).astype(int)
        logging.info("Feature engineering completed: 'is_luxury' column created.")

        df["premium_view"] = df["waterfront"] + df["view"]
        logging.info("Feature engineering completed: 'premium_view' column created.")
        logging.info("Data preprocessing completed successfully.")

        return df
    except Exception as e:
        logging.error(f"Error occurred during data preprocessing: {e}")
        raise

def save_data(data , file_path):
    try:
        """Save the preprocessed data as csv file to the dir"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        X_train_path = os.path.join("data", "processed", "X_train.csv")
        X_test_path = os.path.join("data", "processed", "X_test.csv")
        logging.info("Starting data preprocessing for training and testing sets...")

        X_train_preprocessed = data_preprocessing(X_train_path)
        X_test_preprocessed = data_preprocessing(X_test_path)
        logging.info("Data preprocessing completed for both training and testing sets.")

        save_data(X_train_preprocessed, os.path.join("data", "preprocessed", "X_train_preprocessed.csv"))
        save_data(X_test_preprocessed, os.path.join("data", "preprocessed", "X_test_preprocessed.csv"))
        logging.info("Preprocessed data saved successfully for both training and testing sets.")
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise

if __name__ =="__main__":
    main()