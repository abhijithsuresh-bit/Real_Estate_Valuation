import pandas as pd
import numpy as np
import logging
import yaml
from xgboost import XGBRegressor
import os
import pickle

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    with open("params.yaml", "r")as f:
        params = yaml.safe_load(f)
        logging.info("Successfully loaded parameters from params.yaml")
except Exception as e:
    logging.error(f"Error occurred while loading parameters: {e}")
    raise

def model_building(X_train , y_train , model_params , file_path):
    """"Building model using XGBoost Regressor based on parameters defined in params.yaml"""
    try:
        X_train = pd.read_csv(X_train)
        y_train = pd.read_csv(y_train)
        model = XGBRegressor(**model_params)
        model.fit(X_train , y_train)
        logging.info("Model training completed successfully.")

        # Save the trained model to a file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Trained model saved successfully at {file_path}.")


    except Exception as e:
        logging.error(f"Error occurred during model building: {e}")
        raise


def main():
    try:
        X_train = os.path.join("data", "preprocessed", "X_train_preprocessed.csv")
        y_train = os.path.join("data", "processed", "y_train.csv")

        model_params = {
            "n_estimators": params["model_training"]["hyperparameters"]["n_estimators"],
            "learning_rate": params["model_training"]["hyperparameters"]["learning_rate"],
            "max_depth": params["model_training"]["hyperparameters"]["max_depth"],
            "subsample": params["model_training"]["hyperparameters"]["subsample"],
            "colsample_bytree": params["model_training"]["hyperparameters"]["colsample_bytree"],
            "min_child_weight": params["model_training"]["hyperparameters"]["min_child_weight"],
            "reg_alpha": params["model_training"]["hyperparameters"]["reg_alpha"],
            "reg_lambda": params["model_training"]["hyperparameters"]["reg_lambda"]
        }

        model_file_path = os.path.join("models", "xgb_model.pkl")
        model_building(X_train , y_train , model_params , model_file_path)

    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise

if __name__ == "__main__":
    main()