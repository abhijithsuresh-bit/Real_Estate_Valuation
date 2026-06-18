import mlflow
import json
import logging
import os

logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

logger = logging.getLogger(__name__)

public_ip = "http://ec2-13-48-85-222.eu-north-1.compute.amazonaws.com:5000"

mlflow.set_tracking_uri(public_ip)

def load_model_info(file_path):
    """ Load the model info from a JSON File"""
    try:
        with open(file_path,"r") as f:
            model_info = json.load(f)
            logging.info("Model info loaded from %s" , file_path)

        return model_info
    except Exception as e:
        logging.error("Unexpected error occured while loadig file %s" ,e)
        raise

def register_model(model_name , model_info:dict):
    """Register model to MLflow registry"""
    try:
        model_uri = f"runs:/{model_info["run_id"]}/{model_info["model_path"]}"

        # register model
        model_version = mlflow.register_model(model_uri , model_name)

        #  Transition the model to 'Staging' stage
        '''Staging in MLflow means:
    Staging = "Ready for testing, not yet in production"'''
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name= model_name,
            version=model_version.version,
            stage= "Staging"
        )
        logging.info(f"Model{model_name} version {model_version.version} registered and transitioned to staging")

    except Exception as e:
        logging.error("Error during model registeration %s ",e)

def main():
    try:
        path = os.path.join("mlflow_evaluation_results" , "mlflow_evaluation_results.json")

        model_info = load_model_info(file_path = path)

        model_name = "Real_Estate_Valuation"
        register_model(model_name , model_info)

        logging.info("Done successfully")

    except Exception as e:
        logging.error("Failed to complete %s",e)
        raise

if __name__ == "__main__":
    main()



