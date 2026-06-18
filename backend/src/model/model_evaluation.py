import pandas as pd
import numpy as np
import logging
import yaml
import os 
import pickle
from sklearn.metrics import mean_squared_error, r2_score , root_mean_squared_error
import json
import mlflow
import mlflow.sklearn

# setup logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

try:
    with open("params.yaml","r") as f:
        params = yaml.safe_load(f)
        logging.info("Successfully loaded parameters from params.yaml")
except Exception as e:  
    logging.error(f"Error occurred while loading parameters: {e}")
    raise

def model_evaluation(X_test_file_path , y_test_file_path , model_file_path):
    try:
        # load the model from the file
        with open(model_file_path,"rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {model_file_path}")

        # load the test data
        X_test = pd.read_csv(X_test_file_path)
        y_test = pd.read_csv(y_test_file_path)
        logging.info(f"Test data loaded successfully. Features shape: {X_test.shape}, Target shape: {y_test.shape}")

        # make predictions
        y_pred = model.predict(X_test)
        logging.info(f"Predictions made successfully. Shape: {y_pred.shape}")

        # calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Model Evaluation Metrics:")
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"Root Mean Squared Error: {rmse}")
        logging.info(f"R-squared: {r2}")

        return {"mse": mse, "rmse": rmse, "r2": r2} 

    except Exception as e:
        logging.error(f"Error occurred while evaluating model: {e}")
        raise


def mlflow_evaluation(metrics, model_file_path , public_ip , model_name , params , evaluation_file_path):
    try:
        mlflow.set_tracking_uri(public_ip)
        mlflow.set_experiment("Real_Estate_Valuation Experiment through MLflow")

        with mlflow.start_run(run_name="Model Evaluation") as run:
            # log a description of the evaluation run
            mlflow.set_tag("mlflow.runName" , model_name + " Evaluation Run")
            mlflow.set_tag("Experiment_type" , "Model Evaluation")
            mlflow.set_tag("Model" , model_name)
            logging.info(f"MLflow run started for model evaluation: {model_name}")

            # add a description of the evaluation run
            mlflow.set_tag("Description" , f"This run evaluates the performance of the {model_name} on the test dataset and logs the evaluation metrics.")
            logging.info(f"MLflow run description set for model evaluation: {model_name}")

            # log the parameters used for the model evaluation
            mlflow.log_param("n_estimators" , params["n_estimators"])
            mlflow.log_param("learning_rate" , params["learning_rate"])
            mlflow.log_param("max_depth" , params["max_depth"])
            mlflow.log_param("subsample" , params["subsample"])
            mlflow.log_param("colsample_bytree" , params["colsample_bytree"])
            mlflow.log_param("min_child_weight" , params["min_child_weight"])
            mlflow.log_param("reg_alpha" , params["reg_alpha"])
            mlflow.log_param("reg_lambda" , params["reg_lambda"])
            logging.info(f"MLflow parameters logged for model evaluation: {model_name}")

            # log the evaluation metrics
            mlflow.log_metric("mse" , metrics["mse"])
            mlflow.log_metric("rmse" , metrics["rmse"])
            mlflow.log_metric("r2" , metrics["r2"])
            logging.info(f"MLflow metrics logged for model evaluation: {model_name}")

            # log the model
            with open(model_file_path, "rb") as f:
                model = pickle.load(f)
            mlflow.sklearn.log_model(model, model_name)
            logging.info(f"MLflow model logged for model evaluation: {model_name}")


            model_uri = mlflow.get_artifact_uri(model_name)
            logging.info(f"MLflow model URI: {model_uri}")

            os.makedirs(os.path.dirname(evaluation_file_path), exist_ok=True)

            mlflow_results = {
                "run_id": run.info.run_id,
                "model_path":model_uri,
                "experiment_id":run.info.experiment_id,
                "model_name": model_name,
                "MSE": metrics['mse'],
                "RMSE":metrics["rmse"],
                "R2_score":metrics["r2"]
            }

            with open(evaluation_file_path,"w") as f:
                json.dump(mlflow_results , f )
            logging.info(f"Saving all the evaluation results in {evaluation_file_path}")

    except Exception as e:
        logging.error(f'An error occurred during MLflow evaluation logging: {e}')
        raise
    

def main():
    try:
        X_test_file_path = os.path.join("data" , "preprocessed" , "X_test_preprocessed.csv")
        y_test_file_path = os.path.join("data","processed" , "y_test.csv")
        model_file_path = os.path.join("models" , "xgb_model.pkl")
        public_ip = "http://ec2-13-48-85-222.eu-north-1.compute.amazonaws.com:5000"

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

        evaluation_file_path = os.path.join("mlflow_evaluation_results" , "mlflow_evaluation_results.json")

        metrics = model_evaluation(X_test_file_path= X_test_file_path , y_test_file_path= y_test_file_path , model_file_path= model_file_path)
        mlflow_evaluation(metrics = metrics ,
                           model_file_path = model_file_path,
                           public_ip = public_ip,
                           params = model_params,
                           evaluation_file_path= evaluation_file_path,
                           model_name= params["model_evaluation"]["model_name"]
                          )
    except Exception as e:
        logging.info(f"Error {e}")
        raise


if __name__ == "__main__":
    main()