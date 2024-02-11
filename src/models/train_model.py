import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from src import logger
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from src.utils.common import read_params,accuracymeasures,get_feat_and_target

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["random_forest"]["max_depth"]
    n_estimators=config["random_forest"]["n_estimators"]

    logger.info("Starting training and evaluation process...")

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    logger.info("Data loading completed.")

################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        logger.info("MLflow setup completed.")

        model = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')

        mlflow.log_params({"max_depth": max_depth, "n_estimators": n_estimators})

        mlflow.log_metrics({"accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1score
                            })
       
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")
        logger.info("Model training and evaluation completed.")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)